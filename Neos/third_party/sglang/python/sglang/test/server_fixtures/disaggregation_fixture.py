import logging
import os
import shlex
import time
import warnings
from urllib.parse import urlparse

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_with_error_check,
)
from sglang.utils import wait_for_http_ready

logger = logging.getLogger(__name__)


class PDDisaggregationServerBase(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        parsed_url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.base_host = parsed_url.hostname
        base_port = str(parsed_url.port)
        cls.lb_port = base_port
        cls.prefill_port = f"{int(base_port) + 100}"
        cls.decode_port = f"{int(base_port) + 200}"
        cls.prefill_url = f"http://{cls.base_host}:{cls.prefill_port}"
        cls.decode_url = f"http://{cls.base_host}:{cls.decode_port}"
        cls.lb_url = f"http://{cls.base_host}:{cls.lb_port}"
        print(f"{cls.base_host=} {cls.lb_port=} {cls.prefill_port=} {cls.decode_port=}")
        cls.process_lb, cls.process_decode, cls.process_prefill = None, None, None

        
        if is_in_ci():
            cls.transfer_backend = ["--disaggregation-transfer-backend", "mooncake"]
            cls.rdma_devices = ["--disaggregation-ib-device", get_rdma_devices_args()]
        else:
            cls.transfer_backend = [
                "--disaggregation-transfer-backend",
                envs.SGLANG_TEST_PD_DISAGG_BACKEND.get(),
            ]
            cls.rdma_devices = [
                "--disaggregation-ib-device",
                envs.SGLANG_TEST_PD_DISAGG_DEVICES.get(),
            ]
            if cls.rdma_devices[1] is None:
                cls.rdma_devices = []
                msg = "No RDMA devices specified for disaggregation test, using default settings."
                warnings.warn(msg)

    @classmethod
    def launch_lb(cls):
        lb_command = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--mini-lb",  
            "--prefill",
            cls.prefill_url,
            "--decode",
            cls.decode_url,
            "--host",
            cls.base_host,
            "--port",
            cls.lb_port,
        ]
        print("Starting load balancer:", shlex.join(lb_command))
        cls.process_lb = popen_with_error_check(lb_command)
        cls.wait_server_ready(cls.lb_url + "/health", process=cls.process_lb)

    @classmethod
    def wait_server_ready(
        cls, url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, process=None
    ):
        wait_for_http_ready(url=url, timeout=timeout, process=process)
        print(f"Server {url} is ready")

    @classmethod
    def tearDownClass(cls):
        for process in [cls.process_lb, cls.process_decode, cls.process_prefill]:
            if process:
                try:
                    kill_process_tree(process.pid)
                except Exception as e:
                    print(f"Error killing process {process.pid}: {e}")

        
        time.sleep(5)


def _get_available_ib_devices():
    """Auto-detect available high-speed RDMA devices from sysfs.

    Filters for devices that are:
    1. Not Ethernet NICs (excludes devices with 'eth' in the name like mlx5_eth0)
    2. Active (port state)
    3. High-speed (rate >= 100 Gbps to exclude regular Ethernet NICs)
    """
    ib_sysfs_path = "/sys/class/infiniband"
    if not os.path.isdir(ib_sysfs_path):
        logger.warning("IB sysfs path %s does not exist", ib_sysfs_path)
        return None

    all_devices = sorted(os.listdir(ib_sysfs_path))
    logger.warning("All IB devices in sysfs: %s", all_devices)

    devices = []
    for dev in all_devices:
        
        port_path = os.path.join(ib_sysfs_path, dev, "ports", "1")
        if not os.path.isdir(port_path):
            logger.warning("Device %s: SKIPPED (no port 1)", dev)
            continue

        
        state = "unknown"
        rate = -1
        state_file = os.path.join(port_path, "state")
        rate_file = os.path.join(port_path, "rate")

        try:
            with open(state_file) as f:
                state = f.read().strip()
        except (OSError, IOError):
            pass

        try:
            with open(rate_file) as f:
                rate_str = f.read().strip()
                rate = int(rate_str.split()[0])
        except (OSError, IOError, ValueError, IndexError):
            pass

        
        logger.warning(
            "Device %s: state=%s, rate=%d Gbps, has_eth_in_name=%s",
            dev,
            state,
            rate,
            "eth" in dev.lower(),
        )

        
        
        if "eth" in dev.lower():
            logger.warning("Device %s: SKIPPED (contains 'eth' in name)", dev)
            continue

        
        
        if "ACTIVE" not in state.upper():
            logger.warning("Device %s: SKIPPED (state=%s)", dev, state)
            continue

        
        if rate >= 0 and rate < 100:  
            logger.warning("Device %s: SKIPPED (rate=%d Gbps)", dev, rate)
            continue

        devices.append(dev)
        logger.warning("Device %s: INCLUDED", dev)

    logger.warning("Filtered IB devices: %s (count=%d)", devices, len(devices))
    return devices if devices else None


def get_rdma_devices_args():
    def _parse_list_env(var_name: str):
        val = os.getenv(var_name)
        if not val:
            return None
        items = [x.strip() for x in val.split(",") if x.strip()]
        return items or None

    def _pick_default_pair(rdma_all_devices):
        return [rdma_all_devices[0], rdma_all_devices[len(rdma_all_devices) // 2]]

    
    rdma_all_devices = (
        _parse_list_env("SGLANG_CI_RDMA_ALL_DEVICES")
        or _get_available_ib_devices()
        or [f"mlx5_roce{i}" for i in range(8)]
    )
    logger.warning("Resolved rdma_all_devices=%s", rdma_all_devices)

    n_rdma = len(rdma_all_devices)

    
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if not cuda_visible_devices:
        warnings.warn("CUDA_VISIBLE_DEVICES is not set. Using default RDMA devices.")
        return ",".join(_pick_default_pair(rdma_all_devices))

    try:
        
        gpu_indices = [
            int(idx.strip()) for idx in cuda_visible_devices.split(",") if idx.strip()
        ]
        if not gpu_indices or len(gpu_indices) > 4:
            return ",".join(_pick_default_pair(rdma_all_devices))
    except ValueError:
        warnings.warn(f"Invalid CUDA_VISIBLE_DEVICES format: {cuda_visible_devices}")
        return ",".join(_pick_default_pair(rdma_all_devices))

    
    base_rdma_group = (min(gpu_indices) // 4) * 4
    for gpu_idx in gpu_indices:
        if not (base_rdma_group <= gpu_idx < base_rdma_group + 4):
            warnings.warn(
                f"GPU index {gpu_idx} is outside expected group "
                f"{base_rdma_group}-{base_rdma_group+3}"
            )

    
    
    try:
        import torch

        total_gpus = torch.cuda.device_count()
    except Exception:
        total_gpus = 8  

    
    if total_gpus == 0:
        total_gpus = 8
    if n_rdma > total_gpus:
        logger.warning(
            "More RDMA devices (%d) than GPUs (%d), using first and middle device",
            n_rdma,
            total_gpus,
        )
        return ",".join(_pick_default_pair(rdma_all_devices))

    
    gpus_per_rdma = max(1, total_gpus // n_rdma)
    logger.warning(
        "GPU-to-RDMA mapping: total_gpus=%d, n_rdma=%d, gpus_per_rdma=%d",
        total_gpus,
        n_rdma,
        gpus_per_rdma,
    )

    rdma_devices = []
    for gpu_idx in gpu_indices:
        nic_index = min(gpu_idx // gpus_per_rdma, n_rdma - 1)
        rdma_devices.append(rdma_all_devices[nic_index])

    if not rdma_devices:
        return ",".join(_pick_default_pair(rdma_all_devices))

    return ",".join(rdma_devices)

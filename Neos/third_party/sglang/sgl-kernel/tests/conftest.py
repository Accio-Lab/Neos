import pytest
import torch





@pytest.fixture(autouse=True)
def reset_torch_defaults():
    orig_default_device = torch.get_default_device()
    orig_default_dtype = torch.get_default_dtype()
    yield
    torch.set_default_dtype(orig_default_dtype)
    torch.set_default_device(orig_default_device)

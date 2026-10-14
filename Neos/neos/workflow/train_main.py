from __future__ import annotations
import json
import os
from pathlib import Path
import hydra
from omegaconf import open_dict
from verl.trainer.main_ppo import TaskRunnerV1, run_ppo, validate_config
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from neos.workflow.models import model_family_and_native_tool_parser, official_chat_template
from neos.workflow.image_codec import codec_config
from neos.workflow.protocol import NEOS_PROTOCOL_VERSION, NEOS_SYSTEM_PROMPT, MAX_ASSISTANT_TURNS, MAX_CONTEXT_TOKENS, MAX_TOOL_RESPONSE_LENGTH
from neos.tools.online_client import OnlineSearchClient
_VERL_CONFIG_DIR = str(Path(__file__).resolve().parents[2] / 'verl' / 'verl' / 'trainer' / 'config')
_TOOL_CONFIG_PATH = (Path(__file__).resolve().parents[2] / 'configs' / 'neos' / 'tool_config.yaml').resolve()
_AGENT_LOOP_CONFIG_PATH = (Path(__file__).resolve().parents[2] / 'configs' / 'neos' / 'agent_loop.yaml').resolve()
_JUDGE_REWARD_PATH = (Path(__file__).resolve().parent / 'rl_reward.py').resolve()

def _paths(value) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        values = [value]
    else:
        values = list(value)
    return [Path(str(item)).expanduser().resolve() for item in values]

def _validate_interaction_contract(config, action_parser: str) -> None:
    multi_turn = config.actor_rollout_ref.rollout.multi_turn
    rollout = config.actor_rollout_ref.rollout
    expected_context_tokens = int(os.getenv('NEOS_RL_EXPECTED_CONTEXT_TOKENS', str(MAX_CONTEXT_TOKENS)))
    expected_max_assistant_turns = int(os.getenv('NEOS_RL_EXPECTED_MAX_ASSISTANT_TURNS', str(MAX_ASSISTANT_TURNS)))
    expected = {'multi_turn.enable': (bool(multi_turn.enable), True), 'multi_turn.format': (str(multi_turn.format), action_parser), 'multi_turn.max_parallel_calls': (int(multi_turn.max_parallel_calls), 1), 'multi_turn.max_assistant_turns': (int(multi_turn.max_assistant_turns), expected_max_assistant_turns), 'multi_turn.max_tool_response_length': (int(multi_turn.max_tool_response_length), MAX_TOOL_RESPONSE_LENGTH), 'rollout.max_model_len': (int(rollout.max_model_len), expected_context_tokens), 'rollout.mode': (str(rollout.mode), 'async'), 'rollout.calculate_log_probs': (bool(rollout.calculate_log_probs), True), 'rollout_correction.bypass_mode': (bool(config.algorithm.rollout_correction.bypass_mode), True), 'rollout.agent.default_agent_loop': (str(rollout.agent.default_agent_loop), 'neos_agent')}
    mismatches = {key: {'actual': actual, 'expected': wanted} for key, (actual, wanted) in expected.items() if actual != wanted}
    configured_tool_path = Path(str(multi_turn.tool_config_path)).expanduser().resolve()
    if configured_tool_path != _TOOL_CONFIG_PATH:
        mismatches['multi_turn.tool_config_path'] = {'actual': str(configured_tool_path), 'expected': str(_TOOL_CONFIG_PATH)}
    configured_agent_path = Path(str(rollout.agent.agent_loop_config_path)).expanduser().resolve()
    if configured_agent_path != _AGENT_LOOP_CONFIG_PATH:
        mismatches['rollout.agent.agent_loop_config_path'] = {'actual': str(configured_agent_path), 'expected': str(_AGENT_LOOP_CONFIG_PATH)}
    reward_path = Path(str(config.reward.custom_reward_function.path)).expanduser().resolve()
    if reward_path != _JUDGE_REWARD_PATH:
        mismatches['reward.custom_reward_function.path'] = {'actual': str(reward_path), 'expected': str(_JUDGE_REWARD_PATH)}
    if str(config.reward.custom_reward_function.name) != 'compute_score':
        mismatches['reward.custom_reward_function.name'] = {'actual': str(config.reward.custom_reward_function.name), 'expected': 'compute_score'}
    if str(config.reward.reward_manager.name) != 'rate_limited':
        mismatches['reward.reward_manager.name'] = {'actual': str(config.reward.reward_manager.name), 'expected': 'rate_limited'}
    if mismatches:
        raise RuntimeError('Neos interaction-contract mismatch: ' + json.dumps(mismatches, ensure_ascii=False, sort_keys=True))

@hydra.main(config_path=_VERL_CONFIG_DIR, config_name='ppo_trainer', version_base=None)
def main(config) -> None:
    model_path = Path(str(config.actor_rollout_ref.model.path)).expanduser().resolve()
    model_family, action_parser = model_family_and_native_tool_parser(model_path)
    template = official_chat_template(model_path)
    with open_dict(config):
        config.actor_rollout_ref.model.custom_chat_template = template
    _validate_interaction_contract(config, action_parser)
    search_client = OnlineSearchClient()
    search_client.require_interfaces()
    online = search_client.config_status()
    allow_degraded = os.getenv('NEOS_ALLOW_MISSING_IMAGE_UPLOAD', '').lower() in {'1', 'true', 'yes'}
    if not online['local_reverse_image_ready'] and (not allow_degraded):
        raise RuntimeError('Neos training requires local-image Google Lens staging. Configure the existing Neos OSS adapter.')
    manifest = {'tool_protocol': NEOS_PROTOCOL_VERSION, 'context_image_codec': codec_config(), 'model_family': model_family, 'verl_action_parser': action_parser, 'mei_vlm_model': os.getenv('NEOS_MEI_MODEL_SPEC', 'local/neos-mei'), 'judge_model': os.getenv('NEOS_JUDGE_MODEL', 'qwen3.7-plus'), 'reward': 'evaluation-llm-judge-binary-answer-correctness', 'agent_loop': 'neos.workflow.rl_agent_loop.NeosAgentLoop', 'rollout_mode': str(config.actor_rollout_ref.rollout.mode), 'rollout_context_tokens': int(config.actor_rollout_ref.rollout.max_model_len), 'image_upload_provider': online['image_upload_provider'], 'local_reverse_image_ready': online['local_reverse_image_ready'], 'prompt_datasets': [str(p) for p in _paths(config.data.train_files)]}
    checkpoint_root = config.trainer.default_local_dir
    if checkpoint_root:
        root = Path(str(checkpoint_root)).resolve()
        root.mkdir(parents=True, exist_ok=True)
        path = root / 'neos_tool_manifest.json'
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    validate_config(config=config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
    if not config.trainer.use_v1:
        raise RuntimeError('Neos requires verl v1 trainer')
    if os.getenv('NEOS_TRAIN_PREFLIGHT_ONLY', '').lower() in {'1', 'true', 'yes'}:
        print('Neos training preflight passed; Ray/GPU launch skipped.')
        return
    run_ppo(config, task_runner_class=TaskRunnerV1)
if __name__ == '__main__':
    main()

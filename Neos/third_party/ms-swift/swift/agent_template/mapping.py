# Copyright (c) ModelScope Contributors. All rights reserved.
from .deepseek_v3_1 import DeepSeekV31AgentTemplate
from .deepseek_v4 import DeepSeekV4AgentTemplate
from .extra import ReactGRPOAgentTemplate
from .gemma4 import Gemma4AgentTemplate
from .glm4 import (ChatGLM4AgentTemplate, GLM4_5AgentTemplate, GLM4_7AgentTemplate, GLM4AgentTemplate,
                   GLM5_1AgentTemplate)
from .hermes import HermesAgentTemplate, HunyuanHermesAgentTemplate
from .hy_v3 import HyV3AgentTemplate, HyV3PreviewAgentTemplate
from .kimi_k3 import KimiK3AgentTemplate
from .kimi_k25 import KimiK25AgentTemplate
from .llama import Llama3AgentTemplate, Llama4AgentTemplate
from .minicpm5 import MiniCPM5AgentTemplate
from .minimax_m2 import MinimaxM2AgentTemplate
from .minimax_m3 import MinimaxM3AgentTemplate
from .mistral import MistralAgentTemplate
from .qwen import QwenEnAgentTemplate, QwenEnParallelAgentTemplate, QwenZhAgentTemplate, QwenZhParallelAgentTemplate
from .qwen3_coder import Qwen3_5AgentTemplate, Qwen3CoderAgentTemplate
from .react import ReactEnAgentTemplate, ReactZnAgentTemplate
from .seed_oss import SeedAgentTemplate
from .toolbench import ToolBenchAgentTemplate
from .youtu import YoutuAgentTemplate

agent_template_map = {
    
    'react_en': ReactEnAgentTemplate,
    'react_zh': ReactZnAgentTemplate,
    
    'qwen_en': QwenEnAgentTemplate,
    'qwen_zh': QwenZhAgentTemplate,
    'qwen_en_parallel': QwenEnParallelAgentTemplate,
    'qwen_zh_parallel': QwenZhParallelAgentTemplate,
    'qwen3_coder': Qwen3CoderAgentTemplate,
    'qwen3_5': Qwen3_5AgentTemplate,
    'hermes': HermesAgentTemplate,
    'hunyuan_hermes': HunyuanHermesAgentTemplate,
    'hy_v3_preview': HyV3PreviewAgentTemplate,
    'hy_v3': HyV3AgentTemplate,
    'toolbench': ToolBenchAgentTemplate,  
    'chatglm4': ChatGLM4AgentTemplate,
    'glm4': GLM4AgentTemplate,  
    'glm4_5': GLM4_5AgentTemplate,
    'glm4_7': GLM4_7AgentTemplate,
    'glm5_1': GLM5_1AgentTemplate,
    'llama3': Llama3AgentTemplate,
    'llama4': Llama4AgentTemplate,
    
    'deepseek_v3_1': DeepSeekV31AgentTemplate,
    
    'deepseek_v4': DeepSeekV4AgentTemplate,
    'minimax_m2': MinimaxM2AgentTemplate,
    'minimax_m3': MinimaxM3AgentTemplate,
    'seed_oss': SeedAgentTemplate,
    
    'gemma4': Gemma4AgentTemplate,
    
    'react_grpo': ReactGRPOAgentTemplate,
    'mistral': MistralAgentTemplate,
    'youtu': YoutuAgentTemplate,
    'kimi_k25': KimiK25AgentTemplate,
    'kimi_k3': KimiK3AgentTemplate,
    'minicpm5': MiniCPM5AgentTemplate,
}

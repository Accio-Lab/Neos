from .base import OptimizerCallback
from .galore import GaloreOptimizerCallback
from .lorap import LorapOptimizerCallback
from .multimodal import MultimodalOptimizerCallback
from .muon import MuonOptimizerCallback
from .muonclip import MuonClipOptimizerCallback


optimizers_map = {
    'default': OptimizerCallback,
    'galore': GaloreOptimizerCallback,
    'lorap': LorapOptimizerCallback,
    'muon': MuonOptimizerCallback,
    'muonclip': MuonClipOptimizerCallback,
    'multimodal': MultimodalOptimizerCallback,
}

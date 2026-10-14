# Copyright (c) ModelScope Contributors. All rights reserved.
from .acc import AccMetrics
from .embedding import InfonceMetrics, PairedMetrics
from .nlg import NlgMetrics
from .reranker import RerankerMetrics




eval_metrics_map = {
    'acc': AccMetrics,
    'nlg': NlgMetrics,
    
    'infonce': InfonceMetrics,
    'paired': PairedMetrics,
    
    'reranker': RerankerMetrics,
}

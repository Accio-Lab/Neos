# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np
from transformers import EvalPrediction
from typing import Dict

from swift.utils import get_logger
from .base import EvalMetrics
from .utils import Metric

logger = get_logger()


class RerankerMetrics(EvalMetrics, Metric):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Metric.__init__(self)
        self.add_state('logits', default_factory=list)
        self.add_state('labels', default_factory=list)

    def update(self, logits, labels):
        self.logits.append(logits.cpu().numpy())
        self.labels.append(labels.cpu().numpy())

    def compute(self):
        predictions = np.concatenate(self.logits)
        labels = np.concatenate(self.labels)
        return self._calculate_metrics(predictions, labels)

    def compute_metrics(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        return self._calculate_metrics(eval_prediction.predictions, eval_prediction.label_ids)

    def _calculate_metrics(self, logits, labels):
        """
        Calculate MRR and NDCG metrics for reranker.

        This function first groups the data based on query boundaries (identified by
        positive samples), then calculates MRR and NDCG for each group independently,
        and finally returns the mean across all queries.

        Data format:
        - Each query group starts with a positive sample (label=1) followed by negatives (label=0)
        - Example: [1,0,0,1,0,0,0] represents 2 queries: query1=[1,0,0], query2=[1,0,0,0]

        Args:
            logits: Model output scores [batch_size] (numpy array or can be converted to numpy)
            labels: Binary labels (1 for positive, 0 for negative) [batch_size]

        Returns:
            dict: Dictionary containing MRR and NDCG metrics averaged across all queries
        """
        
        if hasattr(logits, 'numpy'):
            logits = logits.numpy()
        if hasattr(labels, 'numpy'):
            labels = labels.numpy()

        logits = np.array(logits).flatten()
        labels = np.array(labels).flatten()

        
        positive_indices = np.where(labels == 1)[0]

        if len(positive_indices) == 0:
            return {'mrr': 0.0, 'ndcg': 0.0}

        
        query_groups = []
        for i, pos_idx in enumerate(positive_indices):
            
            group_start = pos_idx

            
            if i + 1 < len(positive_indices):
                group_end = positive_indices[i + 1]
            else:
                group_end = len(labels)

            
            query_logits = logits[group_start:group_end]
            query_labels = labels[group_start:group_end]

            query_groups.append((query_logits, query_labels))

        
        mrr_scores = []
        ndcg_scores = []

        for query_idx, (query_logits, query_labels) in enumerate(query_groups):
            
            if len(query_logits) < 2:
                logger.info(f'Query {query_idx}: Skipped (too small: {len(query_logits)} items)')
                continue

            
            if query_labels[0] != 1:
                logger.info(f'Query {query_idx}: Skipped (first sample not positive)')
                continue

            
            ranking = np.argsort(-query_logits)  

            
            pos_rank = np.where(ranking == 0)[0][0] + 1  

            
            mrr = 1.0 / pos_rank
            mrr_scores.append(mrr)

            
            def calculate_ndcg_single_query(relevance_scores, ranking):
                """Calculate NDCG for a single query"""
                
                dcg = 0.0
                for rank_pos, doc_idx in enumerate(ranking):
                    relevance = relevance_scores[doc_idx]
                    dcg += (2**relevance - 1) / np.log2(rank_pos + 2)  

                
                ideal_relevance = np.sort(relevance_scores)[::-1]  
                idcg = 0.0
                for rank_pos, relevance in enumerate(ideal_relevance):
                    idcg += (2**relevance - 1) / np.log2(rank_pos + 2)

                
                if idcg == 0:
                    return 0.0
                return dcg / idcg

            
            relevance_scores = query_labels.astype(float)
            ndcg = calculate_ndcg_single_query(relevance_scores, ranking)
            ndcg_scores.append(ndcg)

        
        if len(mrr_scores) == 0:
            logger.warning('No valid queries found for metric calculation')
            return {'mrr': 0.0, 'ndcg': 0.0}

        mean_mrr = np.mean(mrr_scores)
        mean_ndcg = np.mean(ndcg_scores)

        return {
            'mrr': mean_mrr,
            'ndcg': mean_ndcg,
        }

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import logging
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.utils import Split

logger = logging.getLogger(__name__)


@dataclass
class GPTFIMDatasetConfig(GPTDatasetConfig):
    """Configuration object for Megatron Core GPT FIM datasets"""

    fim_rate: float = None
    """Probability to convert a training sample into a FIM format"""

    fim_spm_rate: float = None
    """Probability that the a FIM sample uses the SPM format over the PSM format"""

    fim_extra_tokens: Dict = None
    """FIM extra tokens. Should consist of prefix, middle, suffix, PAD, and EOD tokens."""

    fim_split_sample: Optional[str] = None
    """String around which to split the sample for FIM"""

    fim_fragment_rate: Optional[float] = None
    """Rate of FIM on each fragment when split_sample is not None"""

    fim_no_prefix: Optional[str] = None
    """Do not apply FIM to fragments that start with this prefix"""


class GPTFIMDataset(GPTDataset):
    """The base GPT dataset

    Args:
        indexed_dataset (IndexedDataset): The IndexedDataset around which to build the
        MegatronDataset

        indexed_indices (np.ndarray): The set of the documents indices to expose

        num_samples (int): The number of samples to draw from the indexed dataset

        index_split (Split): The indexed_indices Split

        config (GPTFIMDatasetConfig): The GPT-specific container for all config sourced parameters
    """

    def __init__(
        self,
        indexed_dataset: IndexedDataset,
        dataset_path: str,
        indexed_indices: np.ndarray,
        num_samples: int,
        index_split: Split,
        config: GPTFIMDatasetConfig,
    ) -> None:
        super().__init__(
            indexed_dataset, dataset_path, indexed_indices, num_samples, index_split, config
        )

        self.np_rng = np.random.RandomState(seed=self.config.random_seed)
        logger.info(f"Initialized FIM RNG with seed = {self.config.random_seed}")
        
        self.fim_rate = self.config.fim_rate
        self.fim_spm_rate = self.config.fim_spm_rate
        self.fragment_fim_rate = self.config.fim_fragment_rate
        fim_split_sample = self.config.fim_split_sample
        self.no_fim_prefix = self.config.fim_no_prefix
        if fim_split_sample:
            fim_split_sample_ids = self.config.tokenizer._tokenizer.tokens_to_ids(fim_split_sample)
            assert isinstance(fim_split_sample_ids, int) or len(fim_split_sample_ids) == 1
            self.fim_split_sample = (
                fim_split_sample_ids
                if isinstance(fim_split_sample_ids, int)
                else fim_split_sample_ids[0]
            )
        else:
            self.fim_split_sample = None

        
        fim_tokens = self.config.fim_extra_tokens
        fim_tokens = [
            fim_tokens["prefix"],
            fim_tokens["middle"],
            fim_tokens["suffix"],
            fim_tokens["pad"],
            fim_tokens["eod"],
        ]
        fim_tokens_ids = self.config.tokenizer._tokenizer.tokens_to_ids(fim_tokens)
        (
            self.prefix_tok_id,
            self.middle_tok_id,
            self.suffix_tok_id,
            self.pad_tok_id,
            self.eod_tok_id,
        ) = fim_tokens_ids

    def _query_document_sample_shuffle_indices(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get the text (token ids) and document ids for a given index

        Args:
            idx (int): The index into the dataset

        Returns:
            Tuple[np.ndarray, np.ndarray]: The text ids and document ids
        """
        
        idx = self.shuffle_index[idx]

        
        doc_index_beg, doc_index_beg_offset = self.sample_index[idx]
        doc_index_end, doc_index_end_offset = self.sample_index[idx + 1]

        document_ids = []
        sample_parts = []

        
        if doc_index_beg == doc_index_end:
            
            document_ids.append(self.document_index[doc_index_beg])

            
            sample_parts.append(
                self.dataset.get(
                    self.document_index[doc_index_beg],
                    offset=doc_index_beg_offset,
                    length=doc_index_end_offset - doc_index_beg_offset + 1,
                )
            )

        
        else:
            for i in range(doc_index_beg, doc_index_end + 1):
                
                document_ids.append(self.document_index[i])

                
                offset = 0 if i > doc_index_beg else doc_index_beg_offset
                length = None if i < doc_index_end else doc_index_end_offset + 1
                sample_parts.append(
                    self.dataset.get(self.document_index[i], offset=offset, length=length)
                )

        sample = np.concatenate(sample_parts)

        sample_len = sample.shape[0]
        segment_breaks = np.argwhere(sample == self.eod_tok_id)

        if segment_breaks.shape != (0, 1):  
            curr_start_position = 0
            new_samples = []
            for loc in np.nditer(segment_breaks):
                
                if loc - curr_start_position > 0:
                    
                    permuted = self._fim_split_and_permute_sequence(sample[curr_start_position:loc])
                    new_samples += [permuted, [self.eod_tok_id]]

                curr_start_position = loc + 1  
            
            permuted = self._fim_split_and_permute_sequence(sample[curr_start_position:])
            new_samples.append(permuted)

            sample = np.concatenate(new_samples)
        else:
            sample = self._fim_split_and_permute_sequence(sample)

        diff = sample.shape[0] - sample_len
        if diff > 0:  
            sample = sample[:sample_len]
        elif diff < 0:  
            sample = np.concatenate([sample, np.full((-1 * diff), self.pad_tok_id)])

        assert sample.shape[0] == sample_len

        return (np.array(sample, dtype=np.int64), np.array(document_ids, dtype=np.int64))

    def _fim_permute_sequence(self, sequence, rate):
        return self._permute(
            sequence,
            rate,
            self.fim_spm_rate,
            self.config.tokenizer,
            truncate_or_pad=False,
            suffix_tok_id=self.suffix_tok_id,
            prefix_tok_id=self.prefix_tok_id,
            middle_tok_id=self.middle_tok_id,
            pad_tok_id=self.pad_tok_id,
            no_fim_prefix=self.no_fim_prefix,
        )

    def _fim_split_and_permute_sequence(self, sequence):
        """
        If self.fim_split_sample is not None, split the sequence.
        Then apply FIM on the fragments, or the whole sequence if self.fim_split_sample is None.
        """
        if self.fim_split_sample is None:
            return self._fim_permute_sequence(sequence, self.fim_rate)
        
        
        
        fragment_breaks = np.argwhere(sequence == self.fim_split_sample)
        if fragment_breaks.shape == (0, 1):
            
            return self._fim_permute_sequence(sequence, self.fim_rate)
        if not self.np_rng.binomial(1, self.fim_rate):
            
            return sequence
        
        curr_start_position = 0
        new_samples = []
        for loc in np.nditer(fragment_breaks):
            if loc - curr_start_position > 0:
                permuted = self._fim_permute_sequence(
                    sequence[curr_start_position:loc], self.fragment_fim_rate
                )
                new_samples += [permuted, [self.fim_split_sample]]
            curr_start_position = loc + 1  
        
        permuted = self._fim_permute_sequence(
            sequence[curr_start_position:], self.fragment_fim_rate
        )
        new_samples.append(permuted)

        return np.concatenate(new_samples)

    def _permute(
        self,
        sample,
        fim_rate,
        fim_spm_rate,
        tokenizer,
        truncate_or_pad=True,
        suffix_tok_id=None,
        prefix_tok_id=None,
        middle_tok_id=None,
        pad_tok_id=None,
        no_fim_prefix=None,
    ):
        """
        Take in a sample (np array w/ size (0,chunklength)) and perform a FIM transformation on it.
        Maintain the same sample length (if transform creates a few extra tokens, drop them).
        """
        if self.np_rng.binomial(1, fim_rate):  

            
            
            
            contents = tokenizer._tokenizer.ids_to_text(sample, remove_special_tokens=True)

            
            if no_fim_prefix is not None and contents.startswith(no_fim_prefix):
                return sample

            try:
                
                
                
                boundaries = list(self.np_rng.randint(low=0, high=len(contents) + 1, size=2))
                boundaries.sort()
            except ValueError as e:
                print(len(contents), contents)
                print(e)
                raise e

            prefix = contents[: boundaries[0]]
            middle = contents[boundaries[0] : boundaries[1]]
            suffix = contents[boundaries[1] :]

            prefix = np.array([*tokenizer._tokenizer.text_to_ids(prefix)], dtype=np.int64)
            middle = np.array([*tokenizer._tokenizer.text_to_ids(middle)], dtype=np.int64)
            suffix = np.array([*tokenizer._tokenizer.text_to_ids(suffix)], dtype=np.int64)

            
            
            
            if truncate_or_pad:
                
                new_length = suffix.shape[0] + prefix.shape[0] + middle.shape[0] + 3
                diff = new_length - sample.shape[0]
                if diff > 0:  
                    if (
                        suffix.shape[0] <= diff
                    ):  
                        return sample
                    suffix = suffix[: suffix.shape[0] - diff]
                elif diff < 0:  
                    suffix = np.concatenate([suffix, np.full((-1 * diff), pad_tok_id)])

            if self.np_rng.binomial(1, fim_spm_rate):
                
                new_sample = np.concatenate(
                    [[prefix_tok_id, suffix_tok_id], suffix, [middle_tok_id], prefix, middle]
                )
            else:
                
                new_sample = np.concatenate(
                    [[prefix_tok_id], prefix, [suffix_tok_id], suffix, [middle_tok_id], middle]
                )

        else:
            
            new_sample = sample

        return new_sample

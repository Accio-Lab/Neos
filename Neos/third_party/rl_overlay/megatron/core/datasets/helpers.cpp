/* Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved. */

/* Helper methods for fast index mapping builds */

#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <math.h>
#include <set>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>

namespace py = pybind11;
using namespace std;

const int32_t LONG_SENTENCE_LEN = 512;


void build_exhaustive_blending_indices(py::array_t<int16_t> &dataset_index, py::array_t<int64_t> &dataset_sample_index, const py::array_t<int64_t> &sizes, const int32_t num_datasets) {
    
                                                                                
                                                                           
    
  auto dataset_index_ptr = dataset_index.mutable_unchecked<1>();
  auto dataset_sample_index_ptr = dataset_sample_index.mutable_unchecked<1>();
  auto sizes_ptr = sizes.unchecked<1>();

  int64_t total_size = 0;
  int64_t dataset_sample_counts[num_datasets];
  std::set<int32_t> dataset_unspent_indices;
  for (int32_t i = 0; i < num_datasets; ++i) {
    total_size += sizes_ptr[i];
    dataset_sample_counts[i] = 0;
    dataset_unspent_indices.insert(i);
  }

                                                                   
  double weights[num_datasets];
  for (int32_t i = 0; i < num_datasets; ++i) {
    weights[i] = sizes_ptr[i] / static_cast<double>(total_size);
  }

  int64_t index_sample = 0;
  while (dataset_unspent_indices.size() > 0) {
    double index_sample_double = std::max(static_cast<double>(index_sample), 1.0);

    int64_t error_argmax = -1;
    double error_max = std::numeric_limits<double>::lowest();

    for (int32_t index_dataset : dataset_unspent_indices) {
      double error = weights[index_dataset] * index_sample_double - static_cast<double>(dataset_sample_counts[index_dataset]);
      if (error > error_max) {
        error_argmax = index_dataset;
        error_max = error;
      }
    }
    assert(error_argmax >= 0);

                            
    dataset_index_ptr[index_sample] = static_cast<int16_t>(error_argmax);
    dataset_sample_index_ptr[index_sample] = dataset_sample_counts[error_argmax];

                                
    dataset_sample_counts[error_argmax] += 1;

    if (sizes_ptr[error_argmax] - static_cast<double>(dataset_sample_counts[error_argmax]) == 0) {
      dataset_unspent_indices.erase(error_argmax);
    }

    index_sample += 1;
  }
}

void build_blending_indices(py::array_t<int16_t> &dataset_index,
                            py::array_t<int64_t> &dataset_sample_index,
                            const py::array_t<double> &weights,
                            const int32_t num_datasets,
                            const int64_t size, const bool verbose)
{
                                                                 
                                        

  if (verbose)
  {
    std::cout << "> building indices for blended datasets ..." << std::endl;
  }

                                               
  auto dataset_index_ptr = dataset_index.mutable_unchecked<1>();
  auto dataset_sample_index_ptr = dataset_sample_index.mutable_unchecked<1>();
  auto weights_ptr = weights.unchecked<1>();

                                                                   
  int64_t current_samples[num_datasets];
  for (int64_t i = 0; i < num_datasets; ++i)
  {
    current_samples[i] = 0;
  }

                     
  for (int64_t sample_idx = 0; sample_idx < size; ++sample_idx)
  {

                                                              
    auto sample_idx_double = std::max(static_cast<double>(sample_idx), 1.0);
    int64_t max_error_index = 0;
    double max_error = weights_ptr[0] * sample_idx_double -
                       static_cast<double>(current_samples[0]);
    for (int64_t dataset_idx = 1; dataset_idx < num_datasets; ++dataset_idx)
    {
      double error = weights_ptr[dataset_idx] * sample_idx_double -
                     static_cast<double>(current_samples[dataset_idx]);
      if (error > max_error)
      {
        max_error = error;
        max_error_index = dataset_idx;
      }
    }

                            
    dataset_index_ptr[sample_idx] = static_cast<int16_t>(max_error_index);
    dataset_sample_index_ptr[sample_idx] = current_samples[max_error_index];

                                
    current_samples[max_error_index] += 1;
  }

               
  if (verbose)
  {
    std::cout << " > sample ratios:" << std::endl;
    for (int64_t dataset_idx = 0; dataset_idx < num_datasets; ++dataset_idx)
    {
      auto ratio = static_cast<double>(current_samples[dataset_idx]) /
                   static_cast<double>(size);
      std::cout << "   dataset " << dataset_idx << ", input: " << weights_ptr[dataset_idx] << ", achieved: " << ratio << std::endl;
    }
  }
}

template <typename T>
py::array_t<T> build_sample_idx(
  const py::array_t<int32_t> &sizes_,
  const py::array_t<int32_t> &document_idx_,
  const int32_t seq_length,
  const int32_t num_epochs,
  const int64_t tokens_per_epoch,
  const bool drop_last_partial_sequence = true,
  const int add_extra_token_to_sequence = 1
){
     
                                                                                                   
                                                                                            
                                                                                                 
                                           
    

                        
  assert(seq_length > 1);
  assert(num_epochs > 0);
  assert(tokens_per_epoch > 1);

                         
  auto sizes = sizes_.unchecked<1>();
  auto document_idx = document_idx_.unchecked<1>();
  
                                                                                                                                      
                                                              
  int64_t num_samples = 0;
  if (drop_last_partial_sequence == true) {
    num_samples = (num_epochs * tokens_per_epoch - add_extra_token_to_sequence) / seq_length;
  }
  else {
    num_samples = ceil(float(num_epochs * tokens_per_epoch - add_extra_token_to_sequence) / seq_length);
  }
  T *sample_idx = new T[2 * (num_samples + 1)];

                           
  int64_t sample_idx_index = 0;
                             
  T document_idx_index = 0;
                                       
  T doc_offset = 0;
                                             
  sample_idx[2 * sample_idx_index] = document_idx_index;
  sample_idx[2 * sample_idx_index + 1] = doc_offset;
  ++sample_idx_index;

  while (sample_idx_index <= num_samples)
  {
                                   
    int32_t remaining_seq_length = seq_length + add_extra_token_to_sequence;
    while (remaining_seq_length != 0)
    {
                                 
      auto document_index = document_idx[document_idx_index];
      auto document_length = sizes[document_index] - doc_offset;
                                            
      remaining_seq_length -= document_length;
                                                                    
                                                                   
                                                               
                                    
      if (remaining_seq_length <= 0)
      {
        doc_offset += (remaining_seq_length + document_length - add_extra_token_to_sequence);
        remaining_seq_length = 0;
      }
      else
      {
                                                                   
        if (document_idx_index == (document_idx_.shape(0) - 1))
        {
                                                                
          assert(sample_idx_index == num_samples);
          doc_offset = sizes[document_idx[document_idx_index]] - add_extra_token_to_sequence;
          break;
        }
        ++document_idx_index;
        doc_offset = 0;
      }
    }
                           
    sample_idx[2 * sample_idx_index] = document_idx_index;
    sample_idx[2 * sample_idx_index + 1] = doc_offset;
    ++sample_idx_index;
  }

                                 
  py::capsule free_when_done(
    sample_idx, 
    [](void *mem_){
	    T *mem = reinterpret_cast<T*>(mem_);
	    delete[] mem;
    }
  );

                            
  const auto byte_size = sizeof(T);
  return py::array_t<T>(
    std::vector<int64_t>{num_samples + 1, 2},         
    {2 * byte_size, byte_size},                                            
    sample_idx,                                                  
    free_when_done                                                     
  );
}

inline int32_t get_target_sample_len(const int32_t short_seq_ratio,
                                     const int32_t max_length,
                                     std::mt19937 &rand32_gen)
{
                               
  if (short_seq_ratio == 0)
  {
    return max_length;
  }
  const auto random_number = rand32_gen();
  if ((random_number % short_seq_ratio) == 0)
  {
    return 2 + random_number % (max_length - 1);
  }
  return max_length;
}

template <typename DocIdx>
py::array build_mapping_impl(const py::array_t<int64_t> &docs_,
                             const py::array_t<int32_t> &sizes_,
                             const int32_t num_epochs,
                             const uint64_t max_num_samples,
                             const int32_t max_seq_length,
                             const double short_seq_prob,
                             const int32_t seed,
                             const bool verbose,
                             const int32_t min_num_sent)
{
                                                                       
                                                                       
                                                       
    

                        
  assert(num_epochs > 0);
  assert(max_seq_length > 1);
  assert(short_seq_prob >= 0.0);
  assert(short_seq_prob <= 1.0);
  assert(seed > 0);

                         
  auto docs = docs_.unchecked<1>();
  auto sizes = sizes_.unchecked<1>();

                                                                              
  int32_t short_seq_ratio = 0;
  if (short_seq_prob > 0)
  {
    short_seq_ratio = static_cast<int32_t>(round(1.0 / short_seq_prob));
  }

  if (verbose)
  {
    const auto sent_start_index = docs[0];
    const auto sent_end_index = docs[docs_.shape(0) - 1];
    const auto num_sentences = sent_end_index - sent_start_index;
    cout << "    using:" << endl
         << std::flush;
    cout << "     number of documents:            " << docs_.shape(0) - 1 << endl
         << std::flush;
    cout << "     sentences range:                [" << sent_start_index << ", " << sent_end_index << ")" << endl
         << std::flush;
    cout << "     total number of sentences:      " << num_sentences << endl
         << std::flush;
    cout << "     number of epochs:               " << num_epochs << endl
         << std::flush;
    cout << "     maximum number of samples:      " << max_num_samples << endl
         << std::flush;
    cout << "     maximum sequence length:        " << max_seq_length << endl
         << std::flush;
    cout << "     short sequence probability:     " << short_seq_prob << endl
         << std::flush;
    cout << "     short sequence ration (1/prob): " << short_seq_ratio << endl
         << std::flush;
    cout << "     seed:                           " << seed << endl
         << std::flush;
  }

                                  
  int64_t num_samples = -1;
  DocIdx *maps = NULL;

                                                                
                                                                      
  bool second = false;
  for (int32_t iteration = 0; iteration < 2; ++iteration)
  {

                                                                
    std::mt19937 rand32_gen(seed);

                                        
    second = (iteration == 1);

                
    uint64_t empty_docs = 0;
    uint64_t one_sent_docs = 0;
    uint64_t long_sent_docs = 0;

                         
    uint64_t map_index = 0;

                      
    for (int32_t epoch = 0; epoch < num_epochs; ++epoch)
    {
      if (map_index >= max_num_samples)
      {
        if (verbose && (!second))
        {
          cout << "    reached " << max_num_samples << " samples after "
               << epoch << " epochs ..." << endl
               << std::flush;
        }
        break;
      }
                           
      for (int32_t doc = 0; doc < (docs.shape(0) - 1); ++doc)
      {

                                                                        
        const auto sent_index_first = docs[doc];
        const auto sent_index_last = docs[doc + 1];

                                                                
                       
        auto prev_start_index = sent_index_first;

                               
        auto num_remain_sent = sent_index_last - sent_index_first;

                           
        if ((epoch == 0) && (!second))
        {
          if (num_remain_sent == 0)
          {
            ++empty_docs;
          }
          if (num_remain_sent == 1)
          {
            ++one_sent_docs;
          }
        }

                                                
        bool contains_long_sentence = false;
        if (num_remain_sent > 1)
        {
          for (auto sent_index = sent_index_first;
               sent_index < sent_index_last; ++sent_index)
          {
            if (sizes[sent_index] > LONG_SENTENCE_LEN)
            {
              if ((epoch == 0) && (!second))
              {
                ++long_sent_docs;
              }
              contains_long_sentence = true;
              break;
            }
          }
        }

                                              
        if ((num_remain_sent >= min_num_sent) && (!contains_long_sentence))
        {

                        
          auto seq_len = int32_t{0};
          auto num_sent = int32_t{0};
          auto target_seq_len = get_target_sample_len(short_seq_ratio,
                                                      max_seq_length,
                                                      rand32_gen);

                                    
          for (auto sent_index = sent_index_first;
               sent_index < sent_index_last; ++sent_index)
          {

                                                    
            seq_len += sizes[sent_index];
            ++num_sent;
            --num_remain_sent;

                                                    
                                                                    
                                                     
                                                          
            if (((seq_len >= target_seq_len) &&
                 (num_remain_sent > 1) &&
                 (num_sent >= min_num_sent)) ||
                (num_remain_sent == 0))
            {

                                    
              if ((3 * map_index + 2) >
                  std::numeric_limits<int64_t>::max())
              {
                cout << "number of samples exceeded maximum "
                     << "allowed by type int64: "
                     << std::numeric_limits<int64_t>::max()
                     << endl;
                throw std::overflow_error("Number of samples");
              }

                                  
              if (second)
              {
                const auto map_index_0 = 3 * map_index;
                maps[map_index_0] = static_cast<DocIdx>(prev_start_index);
                maps[map_index_0 + 1] = static_cast<DocIdx>(sent_index + 1);
                maps[map_index_0 + 2] = static_cast<DocIdx>(target_seq_len);
              }

                                           
              ++map_index;
              prev_start_index = sent_index + 1;
              target_seq_len = get_target_sample_len(short_seq_ratio,
                                                     max_seq_length,
                                                     rand32_gen);
              seq_len = 0;
              num_sent = 0;
            }

          }                                              
        }                                
      }                                                
    }                                                          

    if (!second)
    {
      if (verbose)
      {
        cout << "   number of empty documents: " << empty_docs << endl
             << std::flush;
        cout << "   number of documents with one sentence: " << one_sent_docs << endl
             << std::flush;
        cout << "   number of documents with long sentences: " << long_sent_docs << endl
             << std::flush;
        cout << "   will create mapping for " << map_index << " samples" << endl
             << std::flush;
      }
      assert(maps == NULL);
      assert(num_samples < 0);
      maps = new DocIdx[3 * map_index];
      num_samples = static_cast<int64_t>(map_index);
    }

  }                                                       

             
                                                                   
                            
  std::mt19937_64 rand64_gen(seed + 1);
  for (auto i = (num_samples - 1); i > 0; --i)
  {
    const auto j = static_cast<int64_t>(rand64_gen() % (i + 1));
    const auto i0 = 3 * i;
    const auto j0 = 3 * j;
                   
    swap(maps[i0], maps[j0]);
    swap(maps[i0 + 1], maps[j0 + 1]);
    swap(maps[i0 + 2], maps[j0 + 2]);
  }

                                 
  py::capsule free_when_done(maps, [](void *mem_)
                             {
            DocIdx *mem = reinterpret_cast<DocIdx*>(mem_);
	    delete[] mem; });

                            
  const auto byte_size = sizeof(DocIdx);
  return py::array(std::vector<int64_t>{num_samples, 3},         
                   {3 * byte_size, byte_size},                                        
                   maps,                                                    
                   free_when_done);                                               
}

py::array build_mapping(const py::array_t<int64_t> &docs_,
                        const py::array_t<int> &sizes_,
                        const int num_epochs,
                        const uint64_t max_num_samples,
                        const int max_seq_length,
                        const double short_seq_prob,
                        const int seed,
                        const bool verbose,
                        const int32_t min_num_sent)
{

  if (sizes_.size() > std::numeric_limits<uint32_t>::max())
  {
    if (verbose)
    {
      cout << "    using uint64 for data mapping..." << endl
           << std::flush;
    }
    return build_mapping_impl<uint64_t>(docs_, sizes_, num_epochs,
                                        max_num_samples, max_seq_length,
                                        short_seq_prob, seed, verbose,
                                        min_num_sent);
  }
  else
  {
    if (verbose)
    {
      cout << "    using uint32 for data mapping..." << endl
           << std::flush;
    }
    return build_mapping_impl<uint32_t>(docs_, sizes_, num_epochs,
                                        max_num_samples, max_seq_length,
                                        short_seq_prob, seed, verbose,
                                        min_num_sent);
  }
}

template <typename DocIdx>
py::array build_blocks_mapping_impl(const py::array_t<int64_t> &docs_,
                                    const py::array_t<int32_t> &sizes_,
                                    const py::array_t<int32_t> &titles_sizes_,
                                    const int32_t num_epochs,
                                    const uint64_t max_num_samples,
                                    const int32_t max_seq_length,
                                    const int32_t seed,
                                    const bool verbose,
                                    const bool use_one_sent_blocks)
{
                                                                       
                                                                       
                                                       
    

                        
  assert(num_epochs > 0);
  assert(max_seq_length > 1);
  assert(seed > 0);

                         
  auto docs = docs_.unchecked<1>();
  auto sizes = sizes_.unchecked<1>();
  auto titles_sizes = titles_sizes_.unchecked<1>();

  if (verbose)
  {
    const auto sent_start_index = docs[0];
    const auto sent_end_index = docs[docs_.shape(0) - 1];
    const auto num_sentences = sent_end_index - sent_start_index;
    cout << "    using:" << endl
         << std::flush;
    cout << "     number of documents:            " << docs_.shape(0) - 1 << endl
         << std::flush;
    cout << "     sentences range:                [" << sent_start_index << ", " << sent_end_index << ")" << endl
         << std::flush;
    cout << "     total number of sentences:      " << num_sentences << endl
         << std::flush;
    cout << "     number of epochs:               " << num_epochs << endl
         << std::flush;
    cout << "     maximum number of samples:      " << max_num_samples << endl
         << std::flush;
    cout << "     maximum sequence length:        " << max_seq_length << endl
         << std::flush;
    cout << "     seed:                           " << seed << endl
         << std::flush;
  }

                                 
  int64_t num_samples = -1;
  DocIdx *maps = NULL;

                                              
  int min_num_sent = 2;
  if (use_one_sent_blocks)
  {
    min_num_sent = 1;
  }

                                                                
                                                                      
  bool second = false;
  for (int32_t iteration = 0; iteration < 2; ++iteration)
  {

                                        
    second = (iteration == 1);

                         
    uint64_t map_index = 0;

    uint64_t empty_docs = 0;
    uint64_t one_sent_docs = 0;
    uint64_t long_sent_docs = 0;
                      
    for (int32_t epoch = 0; epoch < num_epochs; ++epoch)
    {
                                       
      int32_t block_id = 0;

      if (map_index >= max_num_samples)
      {
        if (verbose && (!second))
        {
          cout << "    reached " << max_num_samples << " samples after "
               << epoch << " epochs ..." << endl
               << std::flush;
        }
        break;
      }
                           
      for (int32_t doc = 0; doc < (docs.shape(0) - 1); ++doc)
      {

                                                                        
        const auto sent_index_first = docs[doc];
        const auto sent_index_last = docs[doc + 1];
        const auto target_seq_len = max_seq_length - titles_sizes[doc];

                                                                
                       
        auto prev_start_index = sent_index_first;

                               
        auto num_remain_sent = sent_index_last - sent_index_first;

                           
        if ((epoch == 0) && (!second))
        {
          if (num_remain_sent == 0)
          {
            ++empty_docs;
          }
          if (num_remain_sent == 1)
          {
            ++one_sent_docs;
          }
        }
                                                
        bool contains_long_sentence = false;
        if (num_remain_sent >= min_num_sent)
        {
          for (auto sent_index = sent_index_first;
               sent_index < sent_index_last; ++sent_index)
          {
            if (sizes[sent_index] > LONG_SENTENCE_LEN)
            {
              if ((epoch == 0) && (!second))
              {
                ++long_sent_docs;
              }
              contains_long_sentence = true;
              break;
            }
          }
        }
                                                             
        if ((num_remain_sent >= min_num_sent) && (!contains_long_sentence))
        {

                        
          auto seq_len = int32_t{0};
          auto num_sent = int32_t{0};

                                    
          for (auto sent_index = sent_index_first;
               sent_index < sent_index_last; ++sent_index)
          {

                                                    
            seq_len += sizes[sent_index];
            ++num_sent;
            --num_remain_sent;

                                                    
                                                                   
                                                                       
                                                         
            if (((seq_len >= target_seq_len) &&
                 (num_remain_sent >= min_num_sent) &&
                 (num_sent >= min_num_sent)) ||
                (num_remain_sent == 0))
            {

                                  
              if (second)
              {
                const auto map_index_0 = 4 * map_index;
                                                                                               
                                                                                                  
                                                                                   

                maps[map_index_0] = static_cast<DocIdx>(prev_start_index);
                maps[map_index_0 + 1] = static_cast<DocIdx>(sent_index + 1);
                maps[map_index_0 + 2] = static_cast<DocIdx>(doc);
                maps[map_index_0 + 3] = static_cast<DocIdx>(block_id);
              }

                                           
              ++map_index;
              ++block_id;
              prev_start_index = sent_index + 1;
              seq_len = 0;
              num_sent = 0;
            }
          }                                              
        }                                
      }                                                
    }                                                          

    if (!second)
    {
      if (verbose)
      {
        cout << "   number of empty documents: " << empty_docs << endl
             << std::flush;
        cout << "   number of documents with one sentence: " << one_sent_docs << endl
             << std::flush;
        cout << "   number of documents with long sentences: " << long_sent_docs << endl
             << std::flush;
        cout << "   will create mapping for " << map_index << " samples" << endl
             << std::flush;
      }
      assert(maps == NULL);
      assert(num_samples < 0);
      maps = new DocIdx[4 * map_index];
      num_samples = static_cast<int64_t>(map_index);
    }

  }                                                       

             
                                                                   
                            
  std::mt19937_64 rand64_gen(seed + 1);
  for (auto i = (num_samples - 1); i > 0; --i)
  {
    const auto j = static_cast<int64_t>(rand64_gen() % (i + 1));
    const auto i0 = 4 * i;
    const auto j0 = 4 * j;
                   
    swap(maps[i0], maps[j0]);
    swap(maps[i0 + 1], maps[j0 + 1]);
    swap(maps[i0 + 2], maps[j0 + 2]);
    swap(maps[i0 + 3], maps[j0 + 3]);
  }

                                 
  py::capsule free_when_done(maps, [](void *mem_)
                             {
            DocIdx *mem = reinterpret_cast<DocIdx*>(mem_);
	    delete[] mem; });

                            
  const auto byte_size = sizeof(DocIdx);
  return py::array(std::vector<int64_t>{num_samples, 4},         
                   {4 * byte_size, byte_size},                                        
                   maps,                                                    
                   free_when_done);                                               
}

py::array build_blocks_mapping(const py::array_t<int64_t> &docs_,
                               const py::array_t<int> &sizes_,
                               const py::array_t<int> &titles_sizes_,
                               const int num_epochs,
                               const uint64_t max_num_samples,
                               const int max_seq_length,
                               const int seed,
                               const bool verbose,
                               const bool use_one_sent_blocks)
{

  if (sizes_.size() > std::numeric_limits<uint32_t>::max())
  {
    if (verbose)
    {
      cout << "    using uint64 for data mapping..." << endl
           << std::flush;
    }
    return build_blocks_mapping_impl<uint64_t>(docs_, sizes_, titles_sizes_,
                                               num_epochs, max_num_samples, max_seq_length, seed, verbose, use_one_sent_blocks);
  }
  else
  {
    if (verbose)
    {
      cout << "    using uint32 for data mapping..." << endl
           << std::flush;
    }
    return build_blocks_mapping_impl<uint32_t>(docs_, sizes_, titles_sizes_,
                                               num_epochs, max_num_samples, max_seq_length, seed, verbose, use_one_sent_blocks);
  }
}

PYBIND11_MODULE(helpers_cpp, m)
{
  m.def("build_mapping", &build_mapping);
  m.def("build_blocks_mapping", &build_blocks_mapping);
  m.def("build_sample_idx_int32", &build_sample_idx<int32_t>);
  m.def("build_sample_idx_int64", &build_sample_idx<int64_t>);
  m.def("build_blending_indices", &build_blending_indices);
  m.def("build_exhaustive_blending_indices", &build_exhaustive_blending_indices);
}

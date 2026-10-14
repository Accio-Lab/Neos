#pragma once
#include <ATen/core/TensorBody.h>
#include <c10/core/Device.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#include "common.h"

namespace radix_tree_v2 {

struct RadixTree {
 public:
  RadixTree(bool disabled, std::optional<std::size_t> host_size, std::size_t page_size, std::size_t threshold);
  ~RadixTree();

                                                                                    
  RadixTree(const RadixTree&) = delete;
  RadixTree(RadixTree&&) = delete;
  RadixTree& operator=(const RadixTree&) = delete;
  RadixTree& operator=(RadixTree&&) = delete;

                                                                                            
  std::tuple<std::vector<at::Tensor>, std::size_t, NodeHandle, NodeHandle> match_prefix(const token_vec_t& key);
                                                                      
  std::vector<at::Tensor> evict(std::size_t num_tokens);
                              
  void lock_ref(NodeHandle node_id, bool increment                             );
                                                                        
  std::tuple<std::vector<std::tuple<IOTicket, at::Tensor, at::Tensor>>, std::size_t>
  writing_through(const token_vec_t& key, at::Tensor value);
                                                              
  std::tuple<IOTicket, std::vector<at::Tensor>> loading_onboard(NodeHandle host_id, at::Tensor indices);
                                                   
  void commit_writing_through(IOTicket ticket, bool success);
                                                  
  void commit_loading_onboard(IOTicket ticket, bool success);
                                      
  void reset();

                                                                         
  std::size_t evictable_size() const;
                                                   
  std::size_t protected_size() const;
                                               
  std::size_t total_size() const;

                                                 
  void debug_print() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

}                            

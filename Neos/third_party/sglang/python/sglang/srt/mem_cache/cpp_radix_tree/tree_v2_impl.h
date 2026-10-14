#pragma once
#include <c10/util/irange.h>

#include <chrono>
#include <cstddef>
#include <iosfwd>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common.h"
#include "tree_v2.h"
#include "tree_v2_node.h"

namespace radix_tree_v2 {

using node_iterator_t = typename TreeNode::iterator_t;

struct RadixTree::Impl {
 public:
  Impl(bool disabled, bool use_hicache, std::size_t page_size, std::size_t host_size, std::size_t threshold)
      : m_root(             0),
        m_evictable_size(0),
        m_protected_size(0),
        m_cached_vec(),
        m_node_map(),
        m_node_counter(1),                                                   
        disabled(disabled),
        use_hicache(use_hicache),
        page_size(page_size),
        threshold(threshold) {
    _assert(page_size > 0, "Page size must be greater than zero");
    _assert(use_hicache == (host_size > 0), "Hierarchical cache is enabled iff host size > 0");
    m_root.ref_count = 1;                                                  
    m_cached_vec.reserve(page_size);                                       
    m_node_map[m_root.node_id] = &m_root;                        
  }

  TreeNode* split_node(node_iterator_t iterator, std::size_t prefix_length) {
                                                                   
                                                               
    auto old_node_ptr = std::move(iterator->second);
    auto new_node_ptr = std::make_unique<TreeNode>(m_node_counter++);
    auto* old_node = old_node_ptr.get();
    auto* new_node = new_node_ptr.get();
    auto* parent = old_node->parent();
                             
    split_prefix(new_node, old_node, prefix_length);
                                       
    add_child(new_node, std::move(old_node_ptr));
    add_child(parent, std::move(new_node_ptr), iterator);
    m_node_map[new_node->node_id] = new_node;                   
    return new_node;
  }

                     
  TreeNode* create_device_node(TreeNode* parent, token_vec_t vec, at::Tensor indices) {
    auto new_node_ptr = std::make_unique<TreeNode>(m_node_counter++);
    auto new_node = new_node_ptr.get();
    new_node_ptr->_unsafe_tokens() = std::move(vec);
    new_node_ptr->_unsafe_device_indices() = std::move(indices);
    m_evictable_size += new_node_ptr->length();
    add_child(parent, std::move(new_node_ptr));
    m_node_map[new_node->node_id] = new_node;                   
    return new_node;
  }

                     
  void remove_device_node(TreeNode* node) {
    _assert(node->on_gpu_only() && node->ref_count == 0);
    m_evictable_size -= node->length();
    node->parent()->erase_child(get_key(node));
    m_node_map.erase(node->node_id);                        
  }

     
                                                                
                                                                  
                                                                     
                                                             
     
  std::pair<TreeNode*, std::size_t> tree_walk(token_slice key) {
    _assert(key.size() % page_size == 0, "Key should be page-aligned");

    std::size_t total_prefix_length = 0;
    TreeNode* node = &m_root;

    const auto now = std::chrono::steady_clock::now();
    while (key.size() > 0) {
      const auto iterator = node->find_child(get_key(key));
      if (iterator == node->end()) break;

                               
      node = iterator->second.get();

                                                                                       
                                                                   
      const auto prefix_length = align(node->diff_key(key, page_size) + page_size);
      total_prefix_length += prefix_length;

                                                                   
      if (prefix_length < node->length()) {
        return {split_node(iterator, prefix_length), total_prefix_length};
      }

                                                                 
      node->access(now);
      key = key.subspan(prefix_length);
    }

    return {node, total_prefix_length};
  }

  std::vector<TreeNode*> collect_leaves() const {
    std::vector<TreeNode*> leaves;
    std::vector<TreeNode*> stack = {};

    auto process_node = [&](TreeNode* node) {
      if (node->is_leaf()) {
        if (node->ref_count == 0) {
          leaves.push_back(node);
        }
      } else {
        stack.push_back(node);
      }
    };
    for (const auto& [_, child] : m_root) {
      process_node(child.get());
    }
    while (!stack.empty()) {
      const auto node = stack.back();
      stack.pop_back();
      for (const auto& [_, child] : *node) {
        process_node(child.get());
      }
    }
    return leaves;
  }

  std::vector<TreeNode*> collect_leaves_device() const {
                                                                                       
    if (!use_hicache) return collect_leaves();
    std::vector<TreeNode*> leaves;
    std::vector<TreeNode*> stack = {};

    auto process_node = [&](TreeNode* node) {
      if (!node->on_gpu()) return;                                   

      if (node->is_leaf_device()) {
        if (node->ref_count == 0) {
          leaves.push_back(node);
        }
      } else {
        stack.push_back(node);
      }
    };
    for (const auto& [_, child] : m_root) {
      process_node(child.get());
    }
    while (!stack.empty()) {
      const auto node = stack.back();
      stack.pop_back();
      for (const auto& [_, child] : *node) {
        process_node(child.get());
      }
    }

    return leaves;
  }

  void lock_ref(TreeNode* node, bool increment) {
    if (node->is_root()) return;                   
    _assert(node->on_gpu(), "Cannot lock reference on an evicted node");
    if (increment)
      walk_to_root(node, [this](TreeNode* n) {
        if (n->ref_count == 0) {
          m_evictable_size -= n->length();
          m_protected_size += n->length();
        }
        n->ref_count++;
      });
    else
      walk_to_root(node, [this](TreeNode* n) {
        _assert(n->ref_count != 0, "Cannot decrement reference count = zero");
        n->ref_count--;
        if (n->ref_count == 0) {
          m_protected_size -= n->length();
          m_evictable_size += n->length();
        }
      });
  }

  void lock_ref(NodeHandle node_ptr, bool increment) {
    return lock_ref(id2node(node_ptr), increment);
  }

  void lock(TreeNode* node) {
    return lock_ref(node,               true);
  }

  void unlock(TreeNode* node) {
    return lock_ref(node,               false);
  }

  std::size_t total_size() const {
    std::size_t size = 0;
    std::vector<const TreeNode*> stack = {&m_root};
    while (!stack.empty()) {
      auto* node = stack.back();
      stack.pop_back();
      size += node->length();
      for (const auto& [_, child] : *node)
        stack.push_back(child.get());
    }
    return size;
  }

  std::size_t evictable_size() const {
    return m_evictable_size;
  }

  std::size_t protected_size() const {
    return m_protected_size;
  }

  std::size_t align(std::size_t size) const {
    return (size / page_size) * page_size;                       
  }

  TreeNode* id2node(NodeHandle node_id) const {
    const auto iterator = m_node_map.find(node_id);
    _assert(iterator != m_node_map.end(), "Node not found in the map");
    return iterator->second;
  }

  void reset() {
    _assert(m_root.ref_count == 1, "Root node must be protected during reset");
    m_node_counter = 1;                       
    m_root.root_reset();
    m_evictable_size = 0;
    m_protected_size = 0;
    m_node_map.clear();
    m_node_map[m_root.node_id] = &m_root;                           
  }

  void debug_print(std::ostream& os) const;

 private:
                             
  token_vec_t& get_key(token_slice tokens) {
    _assert(tokens.size() >= page_size, "Key should be at least page-sized");
    tokens = tokens.subspan(0, page_size);
    m_cached_vec.assign(tokens.begin(), tokens.end());
    return m_cached_vec;
  }

                                                                         
  token_vec_t& get_key(TreeNode* node) {
    return get_key(node->_unsafe_tokens());
  }

  void add_child(TreeNode* parent, std::unique_ptr<TreeNode>&& child) {
    parent->add_child(get_key(child.get()), std::move(child));
  }

  void add_child(TreeNode* parent, std::unique_ptr<TreeNode>&& child, node_iterator_t it) {
    parent->add_child(it, std::move(child));
  }

  TreeNode m_root;                                                                
  std::size_t m_evictable_size;                                                                              
  std::size_t m_protected_size;                                                                              
  token_vec_t m_cached_vec;                                                                                   
  std::unordered_map<std::size_t, TreeNode*> m_node_map;                              
  std::size_t m_node_counter;                                                    

 public:
                                                            
  const bool disabled;                                                                    
  const bool use_hicache;                                                  
  const std::size_t page_size;                                   
  const std::size_t threshold;                                
};

}                            

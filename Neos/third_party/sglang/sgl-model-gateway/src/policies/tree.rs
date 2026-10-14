use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, VecDeque},
    hash::{BuildHasherDefault, Hasher},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, RwLock,
    },
};

use dashmap::{mapref::entry::Entry, DashMap};
use tracing::debug;

type NodeRef = Arc<Node>;

/// Shard counts for DashMaps to balance concurrency vs allocation overhead.
/// Default DashMap uses num_cpus * 4 shards (e.g., 256 on 64-core machines).
///
/// Root node uses higher shard count since ALL requests pass through it.
/// Other nodes use lower count as traffic diverges through the tree.
///
/// This reduces memory by ~90% vs default while maintaining good concurrency.
const ROOT_SHARD_COUNT: usize = 32;
const NODE_SHARD_COUNT: usize = 8;

/// Create a children DashMap for non-root nodes
#[inline]
fn new_children_map() -> DashMap<char, NodeRef, CharHasherBuilder> {
    DashMap::with_hasher_and_shard_amount(CharHasherBuilder::default(), NODE_SHARD_COUNT)
}

/// Create a tenant access time DashMap for non-root nodes
#[inline]
fn new_tenant_map() -> DashMap<TenantId, u64> {
    DashMap::with_shard_amount(NODE_SHARD_COUNT)
}

/// Interned tenant ID to avoid repeated string allocations.
/// Using Arc<str> allows cheap cloning and comparison.
pub type TenantId = Arc<str>;

/// Result of a prefix match operation, including char counts to avoid recomputation.
#[derive(Debug, Clone)]
pub struct PrefixMatchResult {
    /// The tenant that owns the matched prefix (zero-copy)
    pub tenant: TenantId,
    /// Number of characters matched
    pub matched_char_count: usize,
    /// Total number of characters in the input text
    pub input_char_count: usize,
}

/// A fast identity hasher for single-character keys (used in children DashMap).
/// Since chars have good distribution already, we use identity hashing with mixing.
#[derive(Default)]
struct CharHasher(u64);

impl Hasher for CharHasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline(always)]
    fn write(&mut self, bytes: &[u8]) {
                                                          
        if bytes.len() == 4 {
            let val = u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                                                            
            self.0 = (val as u64).wrapping_mul(0x9E3779B97F4A7C15);
            return;
        }
                                                                    
        for &byte in bytes {
            self.0 = self.0.wrapping_mul(0x100000001b3).wrapping_add(byte as u64);
        }
    }

    #[inline(always)]
    fn write_u32(&mut self, i: u32) {
                                                                           
        self.0 = (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
    }
}

type CharHasherBuilder = BuildHasherDefault<CharHasher>;

/// Advance a string slice by N characters, returning the remaining slice.
/// Returns empty string if n >= char count.
/// Optimized: uses direct byte slicing for ASCII, falls back to char_indices for UTF-8.
#[inline]
fn advance_by_chars(s: &str, n: usize) -> &str {
    if n == 0 {
        return s;
    }
    if n >= s.len() {
        return "";
    }
                                                                       
    let bytes = s.as_bytes();
    if bytes[..n].is_ascii() {
                                                                                 
        return &s[n..];
    }
                                                       
    s.char_indices()
        .nth(n)
        .map(|(idx, _)| &s[idx..])
        .unwrap_or("")
}

/// Get the first N characters of a string as a new String.
/// More efficient than chars().take(n).collect() for known bounds.
#[inline]
fn take_chars(s: &str, n: usize) -> String {
    if n == 0 {
        return String::new();
    }
    s.char_indices()
        .nth(n)
        .map(|(idx, _)| s[..idx].to_string())
        .unwrap_or_else(|| s.to_string())
}

/// Node text with cached character count to avoid repeated O(n) chars().count() calls.
#[derive(Debug)]
struct NodeText {
    /// The actual text stored in this node
    text: String,
    /// Cached character count (UTF-8 chars, not bytes)
    char_count: usize,
}

impl NodeText {
    #[inline]
    fn new(text: String) -> Self {
        let char_count = text.chars().count();
        Self { text, char_count }
    }

    #[inline]
    fn empty() -> Self {
        Self {
            text: String::new(),
            char_count: 0,
        }
    }

    #[inline]
    fn char_count(&self) -> usize {
        self.char_count
    }

    #[inline]
    fn as_str(&self) -> &str {
        &self.text
    }

    #[inline]
    fn first_char(&self) -> Option<char> {
        self.text.chars().next()
    }

    /// Split the text at a character boundary, returning the prefix and suffix.
    /// This is more efficient than slice_by_chars as it computes both at once.
    #[inline]
    fn split_at_char(&self, char_idx: usize) -> (NodeText, NodeText) {
        if char_idx == 0 {
            return (NodeText::empty(), self.clone_text());
        }
        if char_idx >= self.char_count {
            return (self.clone_text(), NodeText::empty());
        }

                                                     
        let byte_idx = self
            .text
            .char_indices()
            .nth(char_idx)
            .map(|(i, _)| i)
            .unwrap_or(self.text.len());

        let prefix = NodeText {
            text: self.text[..byte_idx].to_string(),
            char_count: char_idx,
        };
        let suffix = NodeText {
            text: self.text[byte_idx..].to_string(),
            char_count: self.char_count - char_idx,
        };
        (prefix, suffix)
    }

    #[inline]
    fn clone_text(&self) -> NodeText {
        NodeText {
            text: self.text.clone(),
            char_count: self.char_count,
        }
    }
}

impl Clone for NodeText {
    fn clone(&self) -> Self {
        self.clone_text()
    }
}

/// Global epoch counter for LRU ordering.
/// Uses a simple incrementing counter instead of wall clock time.
///
/// Benefits:
/// - No syscall overhead (vs SystemTime::now())
/// - Smaller memory footprint (u64 vs u128)
/// - Perfectly monotonic (no clock skew issues)
///
/// For LRU eviction, relative ordering is all that matters.
static EPOCH_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Get the next epoch value for LRU timestamp ordering.
/// Uses fetch_add for lock-free, monotonically increasing values.
/// Relaxed ordering is sufficient since we only need eventual consistency
/// for approximate LRU behavior.
#[inline]
fn get_epoch() -> u64 {
    EPOCH_COUNTER.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug)]
struct Node {
    /// Children nodes indexed by first character.
    /// Using custom hasher optimized for char keys.
    children: DashMap<char, NodeRef, CharHasherBuilder>,
    /// Node text with cached character count
    text: RwLock<NodeText>,
    /// Per-tenant last access epoch for LRU ordering. Using TenantId (Arc<str>) for cheap cloning.
    tenant_last_access_time: DashMap<TenantId, u64>,
    /// Parent pointer for upward traversal during timestamp updates
    parent: RwLock<Option<NodeRef>>,
    /// Cached last-accessed tenant for O(1) lookup during prefix match.
    /// Avoids O(shards) DashMap iteration in the common case.
    last_tenant: parking_lot::RwLock<Option<TenantId>>,
}

#[derive(Debug)]
pub struct Tree {
    root: NodeRef,
    /// Per-tenant character count for size tracking. Using TenantId for consistency.
    pub tenant_char_count: DashMap<TenantId, usize>,
}

               

struct EvictionEntry {
    timestamp: u64,
    tenant: TenantId,
    node: NodeRef,
}

impl Eq for EvictionEntry {}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for EvictionEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.timestamp.cmp(&other.timestamp))
    }
}

impl Ord for EvictionEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.timestamp.cmp(&other.timestamp)
    }
}

impl PartialEq for EvictionEntry {
    fn eq(&self, other: &Self) -> bool {
        self.timestamp == other.timestamp
    }
}

                      
                                                                                                                                                              
                                      

/// Count matching prefix characters between two strings.
/// Returns the number of characters that match from the start.
/// Optimized: uses fast byte comparison for ASCII, falls back to char iteration for UTF-8.
#[inline]
fn shared_prefix_count(a: &str, b: &str) -> usize {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();

                                                                                 
    let common_byte_len = a_bytes
        .iter()
        .zip(b_bytes)
        .position(|(&a_byte, &b_byte)| a_byte != b_byte)
        .unwrap_or_else(|| a_bytes.len().min(b_bytes.len()));

                                                                       
                                                                       
    if a_bytes[..common_byte_len].is_ascii() {
        common_byte_len
    } else {
        shared_prefix_count_chars(a, b)
    }
}

/// Fallback char-by-char comparison for strings with non-ASCII characters.
#[inline]
fn shared_prefix_count_chars(a: &str, b: &str) -> usize {
    a.chars()
        .zip(b.chars())
        .take_while(|(a_char, b_char)| a_char == b_char)
        .count()
}

/// Intern a tenant string into an Arc<str> for efficient storage and comparison.
#[inline]
fn intern_tenant(tenant: &str) -> TenantId {
    Arc::from(tenant)
}

impl Default for Tree {
    fn default() -> Self {
        Self::new()
    }
}

impl Tree {
      
                                       

                                                                             
                                                           
                                                    

                  
                                                                             
                                                                     
                                                  
                                                     
      

    pub fn new() -> Self {
        Tree {
                                                                              
            root: Arc::new(Node {
                children: DashMap::with_hasher_and_shard_amount(
                    CharHasherBuilder::default(),
                    ROOT_SHARD_COUNT,
                ),
                text: RwLock::new(NodeText::empty()),
                tenant_last_access_time: DashMap::with_shard_amount(ROOT_SHARD_COUNT),
                parent: RwLock::new(None),
                last_tenant: parking_lot::RwLock::new(None),
            }),
            tenant_char_count: DashMap::with_shard_amount(ROOT_SHARD_COUNT),
        }
    }

    pub fn insert(&self, text: &str, tenant: &str) {
                                                  
                                                                  

                                              
        let tenant_id = intern_tenant(tenant);

                                                                                        
        self.root
            .tenant_last_access_time
            .entry(Arc::clone(&tenant_id))
            .or_insert(0);

        self.tenant_char_count
            .entry(Arc::clone(&tenant_id))
            .or_insert(0);

                                                                 
        let mut remaining = text;
        let mut prev = Arc::clone(&self.root);

                                                            
                                                                          
        enum InsertStep {
            Done,
            Continue {
                next_prev: NodeRef,
                advance_chars: usize,
            },
        }

        while !remaining.is_empty() {
            let first_char = remaining.chars().next().unwrap();

                                                                                               
            let step = match prev.children.entry(first_char) {
                Entry::Vacant(entry) => {
                                                                                        
                                                                                         
                    let remaining_char_count = remaining.chars().count();
                    let epoch = get_epoch();

                    let new_node = Arc::new(Node {
                        children: new_children_map(),
                        text: RwLock::new(NodeText::new(remaining.to_string())),
                        tenant_last_access_time: new_tenant_map(),
                        parent: RwLock::new(Some(Arc::clone(&prev))),
                        last_tenant: parking_lot::RwLock::new(Some(Arc::clone(&tenant_id))),
                    });

                                                                        
                    self.tenant_char_count
                        .entry(Arc::clone(&tenant_id))
                        .and_modify(|count| *count += remaining_char_count)
                        .or_insert(remaining_char_count);
                    new_node
                        .tenant_last_access_time
                        .insert(Arc::clone(&tenant_id), epoch);

                    entry.insert(new_node);
                    InsertStep::Done
                }

                Entry::Occupied(mut entry) => {
                    let matched_node = entry.get().clone();

                    let matched_node_text = matched_node.text.read().unwrap();
                    let matched_node_text_count = matched_node_text.char_count();
                    let matched_node_text_str = matched_node_text.as_str();

                                                                 
                    let shared_count = shared_prefix_count(remaining, matched_node_text_str);

                    if shared_count < matched_node_text_count {
                                                 
                        let (matched_text, contracted_text) =
                            matched_node_text.split_at_char(shared_count);
                        let matched_text_count = shared_count;

                                                                  
                        drop(matched_node_text);

                        let new_node = Arc::new(Node {
                            text: RwLock::new(matched_text),
                            children: new_children_map(),
                            parent: RwLock::new(Some(Arc::clone(&prev))),
                            tenant_last_access_time: matched_node.tenant_last_access_time.clone(),
                            last_tenant: parking_lot::RwLock::new(
                                matched_node.last_tenant.read().clone(),
                            ),
                        });

                        let first_new_char = contracted_text.first_char().unwrap();
                        new_node
                            .children
                            .insert(first_new_char, Arc::clone(&matched_node));

                        entry.insert(Arc::clone(&new_node));

                        *matched_node.text.write().unwrap() = contracted_text;
                        *matched_node.parent.write().unwrap() = Some(Arc::clone(&new_node));

                                                                                                   
                                                                                                       
                        if !new_node
                            .tenant_last_access_time
                            .contains_key(tenant_id.as_ref())
                        {
                            self.tenant_char_count
                                .entry(Arc::clone(&tenant_id))
                                .and_modify(|count| *count += matched_text_count)
                                .or_insert(matched_text_count);
                            new_node
                                .tenant_last_access_time
                                .insert(Arc::clone(&tenant_id), 0);
                        }

                        InsertStep::Continue {
                            next_prev: new_node,
                            advance_chars: shared_count,
                        }
                    } else {
                                                                                              
                        drop(matched_node_text);

                                                                         
                        if !matched_node
                            .tenant_last_access_time
                            .contains_key(tenant_id.as_ref())
                        {
                            self.tenant_char_count
                                .entry(Arc::clone(&tenant_id))
                                .and_modify(|count| *count += matched_node_text_count)
                                .or_insert(matched_node_text_count);
                            matched_node
                                .tenant_last_access_time
                                .insert(Arc::clone(&tenant_id), 0);
                        }

                        InsertStep::Continue {
                            next_prev: matched_node,
                            advance_chars: shared_count,
                        }
                    }
                }
            };

                                                               
            match step {
                InsertStep::Done => return,                                               
                InsertStep::Continue {
                    next_prev,
                    advance_chars,
                } => {
                    prev = next_prev;
                    remaining = advance_by_chars(remaining, advance_chars);
                }
            }
        }

                                                                         
                                                
        let epoch = get_epoch();
        prev.tenant_last_access_time
            .insert(Arc::clone(&tenant_id), epoch);
    }

    /// Performs prefix matching and returns detailed result with char counts.
    /// Optimized: no string allocations, deferred char counting.
    pub fn prefix_match_with_counts(&self, text: &str) -> PrefixMatchResult {
        let mut remaining = text;
        let mut matched_chars = 0;
        let mut prev = Arc::clone(&self.root);

        while !remaining.is_empty() {
            let first_char = remaining.chars().next().unwrap();

            let child_node = prev.children.get(&first_char).map(|e| e.value().clone());

            if let Some(matched_node) = child_node {
                let matched_text_guard = matched_node.text.read().unwrap();
                let matched_node_text_count = matched_text_guard.char_count();

                                                             
                let shared_count = shared_prefix_count(remaining, matched_text_guard.as_str());
                drop(matched_text_guard);

                if shared_count == matched_node_text_count {
                                                                                 
                    matched_chars += shared_count;
                    remaining = advance_by_chars(remaining, shared_count);
                    prev = matched_node;
                } else {
                                                                               
                    matched_chars += shared_count;
                    prev = matched_node;
                    break;
                }
            } else {
                                            
                break;
            }
        }

        let curr = prev;

                                                                                             
                                                                                    
        let tenant: TenantId = {
            let cached = curr.last_tenant.read();
            if let Some(ref t) = *cached {
                if curr.tenant_last_access_time.contains_key(t.as_ref()) {
                    Arc::clone(t)
                } else {
                    drop(cached);
                                                                           
                    let t = curr
                        .tenant_last_access_time
                        .iter()
                        .next()
                        .map(|kv| Arc::clone(kv.key()))
                        .unwrap_or_else(|| Arc::from("empty"));
                    *curr.last_tenant.write() = Some(Arc::clone(&t));
                    t
                }
            } else {
                drop(cached);
                                                       
                let t = curr
                    .tenant_last_access_time
                    .iter()
                    .next()
                    .map(|kv| Arc::clone(kv.key()))
                    .unwrap_or_else(|| Arc::from("empty"));
                *curr.last_tenant.write() = Some(Arc::clone(&t));
                t
            }
        };

                                                                                            
                                                                                       
        let epoch = get_epoch();
        if epoch & 0x7 == 0 {
            curr.tenant_last_access_time
                .insert(Arc::clone(&tenant), epoch);
        }

                                                             
                                                                                     
                                                                      
        let input_char_count = text.chars().count();

        PrefixMatchResult {
            tenant,
            matched_char_count: matched_chars,
            input_char_count,
        }
    }

    /// Legacy prefix_match API for backward compatibility.
    /// Note: This computes matched_text which has allocation overhead.
    pub fn prefix_match(&self, text: &str) -> (String, String) {
        let result = self.prefix_match_with_counts(text);
        let matched_text = take_chars(text, result.matched_char_count);
        (matched_text, result.tenant.to_string())
    }

    #[allow(dead_code)]
    pub fn prefix_match_tenant(&self, text: &str, tenant: &str) -> String {
                                                              

                                                      
        let tenant_id = intern_tenant(tenant);

        let mut remaining = text;
        let mut matched_chars = 0;
        let mut prev = Arc::clone(&self.root);

        while !remaining.is_empty() {
            let first_char = remaining.chars().next().unwrap();

            let child_node = prev.children.get(&first_char).map(|e| e.value().clone());

            if let Some(matched_node) = child_node {
                                                                                      
                if !matched_node
                    .tenant_last_access_time
                    .contains_key(tenant_id.as_ref())
                {
                    break;
                }

                let matched_text_guard = matched_node.text.read().unwrap();
                let matched_node_text_count = matched_text_guard.char_count();

                                                             
                let shared_count = shared_prefix_count(remaining, matched_text_guard.as_str());
                drop(matched_text_guard);

                if shared_count == matched_node_text_count {
                                                                                 
                    matched_chars += shared_count;
                    remaining = advance_by_chars(remaining, shared_count);
                    prev = matched_node;
                } else {
                                                                               
                    matched_chars += shared_count;
                    prev = matched_node;
                    break;
                }
            } else {
                                            
                break;
            }
        }

        let curr = prev;

                                                                              
                                                                          
        if curr
            .tenant_last_access_time
            .contains_key(tenant_id.as_ref())
        {
            let epoch = get_epoch();
            curr.tenant_last_access_time
                .insert(Arc::clone(&tenant_id), epoch);
        }

                                                            
        take_chars(text, matched_chars)
    }

    /// Return the list of tenants for which this node is a leaf.
    /// A tenant is a leaf at this node if no children have that tenant.
    fn leaf_of(node: &NodeRef) -> Vec<TenantId> {
        let mut candidates: HashMap<TenantId, bool> = node
            .tenant_last_access_time
            .iter()
            .map(|entry| (Arc::clone(entry.key()), true))
            .collect();

        for child in node.children.iter() {
            for tenant in child.value().tenant_last_access_time.iter() {
                                                                
                candidates.insert(Arc::clone(tenant.key()), false);
            }
        }

        candidates
            .into_iter()
            .filter(|(_, is_leaf)| *is_leaf)
            .map(|(tenant, _)| tenant)
            .collect()
    }

    pub fn evict_tenant_by_size(&self, max_size: usize) {
                                                 
        let mut stack = vec![Arc::clone(&self.root)];
        let mut pq = BinaryHeap::new();

        while let Some(curr) = stack.pop() {
            for child in curr.children.iter() {
                stack.push(Arc::clone(child.value()));
            }

                                           
            for tenant in Tree::leaf_of(&curr) {
                if let Some(timestamp) = curr.tenant_last_access_time.get(tenant.as_ref()) {
                    pq.push(Reverse(EvictionEntry {
                        timestamp: *timestamp,
                        tenant: Arc::clone(&tenant),
                        node: Arc::clone(&curr),
                    }));
                }
            }
        }

        debug!("Before eviction - Used size per tenant:");
        for entry in self.tenant_char_count.iter() {
            debug!("Tenant: {}, Size: {}", entry.key(), entry.value());
        }

                           
        while let Some(Reverse(entry)) = pq.pop() {
            let EvictionEntry { tenant, node, .. } = entry;

            if let Some(used_size) = self.tenant_char_count.get(tenant.as_ref()) {
                if *used_size <= max_size {
                    continue;
                }
            }

                                                                                  
                                                                            
            let is_still_leaf = node.tenant_last_access_time.contains_key(tenant.as_ref())
                && !node.children.iter().any(|child| {
                    child
                        .value()
                        .tenant_last_access_time
                        .contains_key(tenant.as_ref())
                });
            if !is_still_leaf {
                continue;
            }

                                                       
            let node_len = node.text.read().unwrap().char_count();
            self.tenant_char_count
                .entry(Arc::clone(&tenant))
                .and_modify(|count| {
                    *count = count.saturating_sub(node_len);
                });

                                      
            node.tenant_last_access_time.remove(tenant.as_ref());

                                                               
            let parent_opt = node.parent.read().unwrap().clone();

                                 
            if node.children.is_empty() && node.tenant_last_access_time.is_empty() {
                if let Some(ref parent) = parent_opt {
                    if let Some(fc) = node.text.read().unwrap().first_char() {
                        parent.children.remove(&fc);
                    }
                }
            }

                                                                       
                                                                
            if let Some(ref parent) = parent_opt {
                if parent.tenant_last_access_time.contains_key(tenant.as_ref()) {
                    let has_child_with_tenant = parent.children.iter().any(|child| {
                        child
                            .value()
                            .tenant_last_access_time
                            .contains_key(tenant.as_ref())
                    });

                    if !has_child_with_tenant {
                                                                   
                        if let Some(timestamp) = parent.tenant_last_access_time.get(tenant.as_ref())
                        {
                            pq.push(Reverse(EvictionEntry {
                                timestamp: *timestamp,
                                tenant: Arc::clone(&tenant),
                                node: Arc::clone(parent),
                            }));
                        }
                    }
                }
            }
        }

        debug!("After eviction - Used size per tenant:");
        for entry in self.tenant_char_count.iter() {
            debug!("Tenant: {}, Size: {}", entry.key(), entry.value());
        }
    }

    pub fn remove_tenant(&self, tenant: &str) {
                                                      
        let tenant_id = intern_tenant(tenant);

                                                
                                                                        
        let mut stack = vec![Arc::clone(&self.root)];
        let mut queue = VecDeque::new();

        while let Some(curr) = stack.pop() {
            for child in curr.children.iter() {
                stack.push(Arc::clone(child.value()));
            }

                                                          
            if curr
                .tenant_last_access_time
                .contains_key(tenant_id.as_ref())
            {
                let has_child_with_tenant = curr.children.iter().any(|child| {
                    child
                        .value()
                        .tenant_last_access_time
                        .contains_key(tenant_id.as_ref())
                });
                if !has_child_with_tenant {
                    queue.push_back(Arc::clone(&curr));
                }
            }
        }

                                                                                                   
        while let Some(curr) = queue.pop_front() {
                                      
            curr.tenant_last_access_time.remove(tenant_id.as_ref());

                                                               
            let parent_opt = curr.parent.read().unwrap().clone();

                                 
            if curr.children.is_empty() && curr.tenant_last_access_time.is_empty() {
                if let Some(ref parent) = parent_opt {
                    if let Some(fc) = curr.text.read().unwrap().first_char() {
                        parent.children.remove(&fc);
                    }
                }
            }

                                                                       
                                                       
            if let Some(ref parent) = parent_opt {
                if parent
                    .tenant_last_access_time
                    .contains_key(tenant_id.as_ref())
                {
                    let has_child_with_tenant = parent.children.iter().any(|child| {
                        child
                            .value()
                            .tenant_last_access_time
                            .contains_key(tenant_id.as_ref())
                    });

                    if !has_child_with_tenant {
                        queue.push_back(Arc::clone(parent));
                    }
                }
            }
        }

                                                              
        self.tenant_char_count.remove(tenant_id.as_ref());
    }

    #[allow(dead_code)]
    pub fn get_tenant_char_count(&self) -> HashMap<String, usize> {
        self.tenant_char_count
            .iter()
            .map(|entry| (entry.key().to_string(), *entry.value()))
            .collect()
    }

    #[allow(dead_code)]
    pub fn get_used_size_per_tenant(&self) -> HashMap<String, usize> {
                                                                                               

        let mut used_size_per_tenant: HashMap<String, usize> = HashMap::new();
        let mut stack = vec![Arc::clone(&self.root)];

        while let Some(curr) = stack.pop() {
                                                               
            let text_count = curr.text.read().unwrap().char_count();

            for tenant in curr.tenant_last_access_time.iter() {
                let size = used_size_per_tenant
                    .entry(tenant.key().to_string())
                    .or_insert(0);
                *size += text_count;
            }

            for child in curr.children.iter() {
                stack.push(Arc::clone(child.value()));
            }
        }

        used_size_per_tenant
    }

    #[allow(dead_code)]
    fn node_to_string(node: &NodeRef, prefix: &str, is_last: bool) -> String {
        let mut result = String::new();

                                          
        result.push_str(prefix);
        result.push_str(if is_last { "└── " } else { "├── " });

                        
        let node_text = node.text.read().unwrap();
        result.push_str(&format!("'{}' [", node_text.as_str()));

                                                   
        let mut tenant_info = Vec::new();
        for entry in node.tenant_last_access_time.iter() {
            let tenant_id = entry.key();
            let epoch = entry.value();
            tenant_info.push(format!("{} | epoch:{}", tenant_id, epoch));
        }

        result.push_str(&tenant_info.join(", "));
        result.push_str("]\n");

                           
        let children: Vec<_> = node.children.iter().collect();
        let child_count = children.len();

        for (i, entry) in children.iter().enumerate() {
            let is_last_child = i == child_count - 1;
            let new_prefix = format!("{}{}", prefix, if is_last { "    " } else { "│   " });

            result.push_str(&Tree::node_to_string(
                entry.value(),
                &new_prefix,
                is_last_child,
            ));
        }

        result
    }

    #[allow(dead_code)]
    pub fn pretty_print(&self) {
        if self.root.children.is_empty() {
            return;
        }

        let mut result = String::new();
        let children: Vec<_> = self.root.children.iter().collect();
        let child_count = children.len();

        for (i, entry) in children.iter().enumerate() {
            let is_last = i == child_count - 1;
            result.push_str(&Tree::node_to_string(entry.value(), "", is_last));
        }

        println!("{result}");
    }
}

              
#[cfg(test)]
mod tests {
    use std::{
        thread,
        time::{Duration, Instant},
    };

    use rand::{
        distr::{Alphanumeric, SampleString},
        rng as thread_rng, Rng,
    };

    use super::*;

    /// Helper to convert tenant_char_count to HashMap<String, usize> for comparison
    fn get_maintained_counts(tree: &Tree) -> HashMap<String, usize> {
        tree.tenant_char_count
            .iter()
            .map(|entry| (entry.key().to_string(), *entry.value()))
            .collect()
    }

    #[test]
    fn test_tenant_char_count() {
        let tree = Tree::new();

        tree.insert("apple", "tenant1");
        tree.insert("apricot", "tenant1");
        tree.insert("banana", "tenant1");
        tree.insert("amplify", "tenant2");
        tree.insert("application", "tenant2");

        let computed_sizes = tree.get_used_size_per_tenant();
        let maintained_counts = get_maintained_counts(&tree);

        println!("Phase 1 - Maintained vs Computed counts:");
        println!(
            "Maintained: {:?}\nComputed: {:?}",
            maintained_counts, computed_sizes
        );
        assert_eq!(
            maintained_counts, computed_sizes,
            "Phase 1: Initial insertions"
        );

        tree.insert("apartment", "tenant1");
        tree.insert("appetite", "tenant2");
        tree.insert("ball", "tenant1");
        tree.insert("box", "tenant2");

        let computed_sizes = tree.get_used_size_per_tenant();
        let maintained_counts = get_maintained_counts(&tree);

        println!("Phase 2 - Maintained vs Computed counts:");
        println!(
            "Maintained: {:?}\nComputed: {:?}",
            maintained_counts, computed_sizes
        );
        assert_eq!(
            maintained_counts, computed_sizes,
            "Phase 2: Additional insertions"
        );

        tree.insert("zebra", "tenant1");
        tree.insert("zebra", "tenant2");
        tree.insert("zero", "tenant1");
        tree.insert("zero", "tenant2");

        let computed_sizes = tree.get_used_size_per_tenant();
        let maintained_counts = get_maintained_counts(&tree);

        println!("Phase 3 - Maintained vs Computed counts:");
        println!(
            "Maintained: {:?}\nComputed: {:?}",
            maintained_counts, computed_sizes
        );
        assert_eq!(
            maintained_counts, computed_sizes,
            "Phase 3: Overlapping insertions"
        );

        tree.evict_tenant_by_size(10);

        let computed_sizes = tree.get_used_size_per_tenant();
        let maintained_counts = get_maintained_counts(&tree);

        println!("Phase 4 - Maintained vs Computed counts:");
        println!(
            "Maintained: {:?}\nComputed: {:?}",
            maintained_counts, computed_sizes
        );
        assert_eq!(maintained_counts, computed_sizes, "Phase 4: After eviction");
    }

    fn random_string(len: usize) -> String {
        Alphanumeric.sample_string(&mut thread_rng(), len)
    }

    #[test]
    fn test_cold_start() {
        let tree = Tree::new();

        let (matched_text, tenant) = tree.prefix_match("hello");

        assert_eq!(matched_text, "");
        assert_eq!(tenant, "empty");
    }

    #[test]
    fn test_exact_match_seq() {
        let tree = Tree::new();
        tree.insert("hello", "tenant1");
        tree.pretty_print();
        tree.insert("apple", "tenant2");
        tree.pretty_print();
        tree.insert("banana", "tenant3");
        tree.pretty_print();

        let (matched_text, tenant) = tree.prefix_match("hello");
        assert_eq!(matched_text, "hello");
        assert_eq!(tenant, "tenant1");

        let (matched_text, tenant) = tree.prefix_match("apple");
        assert_eq!(matched_text, "apple");
        assert_eq!(tenant, "tenant2");

        let (matched_text, tenant) = tree.prefix_match("banana");
        assert_eq!(matched_text, "banana");
        assert_eq!(tenant, "tenant3");
    }

    #[test]
    fn test_exact_match_concurrent() {
        let tree = Arc::new(Tree::new());

                                     
        let tree_clone = Arc::clone(&tree);

        let texts = ["hello", "apple", "banana"];
        let tenants = ["tenant1", "tenant2", "tenant3"];

        let mut handles = vec![];

        for i in 0..3 {
            let tree_clone = Arc::clone(&tree_clone);
            let text = texts[i];
            let tenant = tenants[i];

            let handle = thread::spawn(move || {
                tree_clone.insert(text, tenant);
            });

            handles.push(handle);
        }

               
        for handle in handles {
            handle.join().unwrap();
        }

                                    
        let mut handles = vec![];

        let tree_clone = Arc::clone(&tree);

        for i in 0..3 {
            let tree_clone = Arc::clone(&tree_clone);
            let text = texts[i];
            let tenant = tenants[i];

            let handle = thread::spawn(move || {
                let (matched_text, matched_tenant) = tree_clone.prefix_match(text);
                assert_eq!(matched_text, text);
                assert_eq!(matched_tenant, tenant);
            });

            handles.push(handle);
        }

               
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_partial_match_concurrent() {
        let tree = Arc::new(Tree::new());

                                     
        let tree_clone = Arc::clone(&tree);

        static TEXTS: [&str; 3] = ["apple", "apabc", "acbdeds"];

        let mut handles = vec![];

        for text in TEXTS.iter() {
            let tree_clone = Arc::clone(&tree_clone);
            let tenant = "tenant0";

            let handle = thread::spawn(move || {
                tree_clone.insert(text, tenant);
            });

            handles.push(handle);
        }

               
        for handle in handles {
            handle.join().unwrap();
        }

                                    
        let mut handles = vec![];

        let tree_clone = Arc::clone(&tree);

        for text in TEXTS.iter() {
            let tree_clone = Arc::clone(&tree_clone);
            let tenant = "tenant0";

            let handle = thread::spawn(move || {
                let (matched_text, matched_tenant) = tree_clone.prefix_match(text);
                assert_eq!(matched_text, *text);
                assert_eq!(matched_tenant, tenant);
            });

            handles.push(handle);
        }

               
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_group_prefix_insert_match_concurrent() {
        static PREFIXES: [&str; 4] = [
            "Clock strikes midnight, I'm still wide awake",
            "Got dreams bigger than these city lights",
            "Time waits for no one, gotta make my move",
            "Started from the bottom, that's no metaphor",
        ];
        let suffixes = [
            "Got too much to prove, ain't got time to lose",
            "History in the making, yeah, you can't erase this",
        ];
        let tree = Arc::new(Tree::new());

        let mut handles = vec![];

        for (i, prefix) in PREFIXES.iter().enumerate() {
            for suffix in suffixes.iter() {
                let tree_clone = Arc::clone(&tree);
                let text = format!("{} {}", prefix, suffix);
                let tenant = format!("tenant{}", i);

                let handle = thread::spawn(move || {
                    tree_clone.insert(&text, &tenant);
                });

                handles.push(handle);
            }
        }

               
        for handle in handles {
            handle.join().unwrap();
        }

        tree.pretty_print();

                                             
        let mut handles = vec![];

        for (i, prefix) in PREFIXES.iter().enumerate() {
            let tree_clone = Arc::clone(&tree);

            let handle = thread::spawn(move || {
                let (matched_text, matched_tenant) = tree_clone.prefix_match(prefix);
                let tenant = format!("tenant{}", i);
                assert_eq!(matched_text, *prefix);
                assert_eq!(matched_tenant, tenant);
            });

            handles.push(handle);
        }

               
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_mixed_concurrent_insert_match() {
                                                                         

        static PREFIXES: [&str; 4] = [
            "Clock strikes midnight, I'm still wide awake",
            "Got dreams bigger than these city lights",
            "Time waits for no one, gotta make my move",
            "Started from the bottom, that's no metaphor",
        ];
        let suffixes = [
            "Got too much to prove, ain't got time to lose",
            "History in the making, yeah, you can't erase this",
        ];
        let tree = Arc::new(Tree::new());

        let mut handles = vec![];

        for (i, prefix) in PREFIXES.iter().enumerate() {
            for suffix in suffixes.iter() {
                let tree_clone = Arc::clone(&tree);
                let text = format!("{} {}", prefix, suffix);
                let tenant = format!("tenant{}", i);

                let handle = thread::spawn(move || {
                    tree_clone.insert(&text, &tenant);
                });

                handles.push(handle);
            }
        }

                                             
        for prefix in PREFIXES.iter() {
            let tree_clone = Arc::clone(&tree);

            let handle = thread::spawn(move || {
                let (_matched_text, _matched_tenant) = tree_clone.prefix_match(prefix);
            });

            handles.push(handle);
        }

               
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_utf8_split_seq() {
                                                                                              
                                                              
        let tree = Arc::new(Tree::new());

        static TEST_PAIRS: [(&str, &str); 3] = [
            ("你好嗎", "tenant1"),
            ("你好喔", "tenant2"),
            ("你心情好嗎", "tenant3"),
        ];

                              
        for (text, tenant) in TEST_PAIRS.iter() {
            tree.insert(text, tenant);
        }

        tree.pretty_print();

        for (text, tenant) in TEST_PAIRS.iter() {
            let (matched_text, matched_tenant) = tree.prefix_match(text);
            assert_eq!(matched_text, *text);
            assert_eq!(matched_tenant, *tenant);
        }
    }

    #[test]
    fn test_utf8_split_concurrent() {
        let tree = Arc::new(Tree::new());

        static TEST_PAIRS: [(&str, &str); 3] = [
            ("你好嗎", "tenant1"),
            ("你好喔", "tenant2"),
            ("你心情好嗎", "tenant3"),
        ];

                                                
        let mut handles = vec![];

        for (text, tenant) in TEST_PAIRS.iter() {
            let tree_clone = Arc::clone(&tree);

            let handle = thread::spawn(move || {
                tree_clone.insert(text, tenant);
            });

            handles.push(handle);
        }

                                              
        for handle in handles {
            handle.join().unwrap();
        }

        tree.pretty_print();

                                               
        let mut handles = vec![];

        for (text, tenant) in TEST_PAIRS.iter() {
            let tree_clone = Arc::clone(&tree);

            let handle = thread::spawn(move || {
                let (matched_text, matched_tenant) = tree_clone.prefix_match(text);
                assert_eq!(matched_text, *text);
                assert_eq!(matched_tenant, *tenant);
            });

            handles.push(handle);
        }

                                           
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_simple_eviction() {
        let tree = Tree::new();
        let max_size = 5;

                                          
        tree.insert("hello", "tenant1");          

        tree.insert("hello", "tenant2");          
        thread::sleep(Duration::from_millis(10));
        tree.insert("world", "tenant2");                                  

        tree.pretty_print();

        let sizes_before = tree.get_used_size_per_tenant();
        assert_eq!(sizes_before.get("tenant1").unwrap(), &5);               
        assert_eq!(sizes_before.get("tenant2").unwrap(), &10);                          

                                                                        
        tree.evict_tenant_by_size(max_size);

        tree.pretty_print();

        let sizes_after = tree.get_used_size_per_tenant();
        assert_eq!(sizes_after.get("tenant1").unwrap(), &5);                       
        assert_eq!(sizes_after.get("tenant2").unwrap(), &5);                        

        let (matched, tenant) = tree.prefix_match("world");
        assert_eq!(matched, "world");
        assert_eq!(tenant, "tenant2");
    }

    #[test]
    fn test_advanced_eviction() {
        let tree = Tree::new();

                                     
        let max_size: usize = 100;

                          
        let prefixes = ["aqwefcisdf", "iajsdfkmade", "kjnzxcvewqe", "iejksduqasd"];

                                              
        for _i in 0..100 {
            for (j, prefix) in prefixes.iter().enumerate() {
                let random_suffix = random_string(10);
                let text = format!("{}{}", prefix, random_suffix);
                let tenant = format!("tenant{}", j + 1);
                tree.insert(&text, &tenant);
            }
        }

                           
        tree.evict_tenant_by_size(max_size);

                                     
        let sizes_after = tree.get_used_size_per_tenant();
        for (tenant, &size) in sizes_after.iter() {
            assert!(
                size <= max_size,
                "Tenant {} exceeds size limit. Current size: {}, Limit: {}",
                tenant,
                size,
                max_size
            );
        }
    }

    #[test]
    fn test_concurrent_operations_with_eviction() {
                                                                                                    

        let tree = Arc::new(Tree::new());
        let mut handles = vec![];
        let test_duration = Duration::from_secs(10);
        let start_time = Instant::now();
        let max_size = 100;                                   

                                
        {
            let tree = Arc::clone(&tree);
            let handle = thread::spawn(move || {
                while start_time.elapsed() < test_duration {
                                   
                    tree.evict_tenant_by_size(max_size);

                                          
                    thread::sleep(Duration::from_secs(5));
                }
            });
            handles.push(handle);
        }

                                 
        for thread_id in 0..4 {
            let tree = Arc::clone(&tree);
            let handle = thread::spawn(move || {
                let mut rng = rand::rng();
                let tenant = format!("tenant{}", thread_id + 1);
                let prefix = format!("prefix{}", thread_id);

                while start_time.elapsed() < test_duration {
                                                                               
                    if rng.random_bool(0.7) {
                                                  
                        let random_len = rng.random_range(3..10);
                        let search_str = format!("{}{}", prefix, random_string(random_len));
                        let (_matched, _) = tree.prefix_match(&search_str);
                    } else {
                                                   
                        let random_len = rng.random_range(5..15);
                        let insert_str = format!("{}{}", prefix, random_string(random_len));
                        tree.insert(&insert_str, &tenant);
                                                                                     
                    }

                                                        
                    thread::sleep(Duration::from_millis(rng.random_range(10..100)));
                }
            });
            handles.push(handle);
        }

                                           
        for handle in handles {
            handle.join().unwrap();
        }

                         
        tree.evict_tenant_by_size(max_size);

                           
        let final_sizes = tree.get_used_size_per_tenant();
        println!("Final sizes after test completion: {:?}", final_sizes);

        for (_, &size) in final_sizes.iter() {
            assert!(
                size <= max_size,
                "Tenant exceeds size limit. Final size: {}, Limit: {}",
                size,
                max_size
            );
        }
    }

    #[test]
    fn test_leaf_of() {
        let tree = Tree::new();

                                                                   
        let leaves_as_strings =
            |leaves: &[TenantId]| -> Vec<String> { leaves.iter().map(|t| t.to_string()).collect() };

                      
        tree.insert("hello", "tenant1");
        let leaves = Tree::leaf_of(&tree.root.children.get(&'h').unwrap());
        assert_eq!(leaves_as_strings(&leaves), vec!["tenant1"]);

                                     
        tree.insert("hello", "tenant2");
        let leaves = Tree::leaf_of(&tree.root.children.get(&'h').unwrap());
        let leaves_str = leaves_as_strings(&leaves);
        assert_eq!(leaves_str.len(), 2);
        assert!(leaves_str.contains(&"tenant1".to_string()));
        assert!(leaves_str.contains(&"tenant2".to_string()));

                        
        tree.insert("hi", "tenant1");
        let leaves = Tree::leaf_of(&tree.root.children.get(&'h').unwrap());
        assert!(leaves.is_empty());
    }

    #[test]
    fn test_get_used_size_per_tenant() {
        let tree = Tree::new();

                        
        tree.insert("hello", "tenant1");
        tree.insert("world", "tenant1");
        let sizes = tree.get_used_size_per_tenant();

        tree.pretty_print();
        println!("{:?}", sizes);
        assert_eq!(sizes.get("tenant1").unwrap(), &10);                     

                                         
        tree.insert("hello", "tenant2");
        tree.insert("help", "tenant2");
        let sizes = tree.get_used_size_per_tenant();

        tree.pretty_print();
        println!("{:?}", sizes);
        assert_eq!(sizes.get("tenant1").unwrap(), &10);
        assert_eq!(sizes.get("tenant2").unwrap(), &6);                 

                           
        tree.insert("你好", "tenant3");
        let sizes = tree.get_used_size_per_tenant();
        tree.pretty_print();
        println!("{:?}", sizes);
        assert_eq!(sizes.get("tenant3").unwrap(), &2);                        

        tree.pretty_print();
    }

    #[test]
    fn test_prefix_match_tenant() {
        let tree = Tree::new();

                                                            
        tree.insert("hello", "tenant1");                  
        tree.insert("hello", "tenant2");                  
        tree.insert("hello world", "tenant2");                           
        tree.insert("help", "tenant1");                     
        tree.insert("helicopter", "tenant2");                           

        assert_eq!(tree.prefix_match_tenant("hello", "tenant1"), "hello");                          
        assert_eq!(tree.prefix_match_tenant("help", "tenant1"), "help");                        
        assert_eq!(tree.prefix_match_tenant("hel", "tenant1"), "hel");                 
        assert_eq!(tree.prefix_match_tenant("hello world", "tenant1"), "hello");                                     
        assert_eq!(tree.prefix_match_tenant("helicopter", "tenant1"), "hel");                                     

        assert_eq!(tree.prefix_match_tenant("hello", "tenant2"), "hello");                          
        assert_eq!(
            tree.prefix_match_tenant("hello world", "tenant2"),
            "hello world"
        );                        
        assert_eq!(
            tree.prefix_match_tenant("helicopter", "tenant2"),
            "helicopter"
        );                        
        assert_eq!(tree.prefix_match_tenant("hel", "tenant2"), "hel");                 
        assert_eq!(tree.prefix_match_tenant("help", "tenant2"), "hel");                                     

        assert_eq!(tree.prefix_match_tenant("hello", "tenant3"), "");                       
        assert_eq!(tree.prefix_match_tenant("help", "tenant3"), "");                       
    }

    #[test]
    fn test_simple_tenant_eviction() {
        let tree = Tree::new();

                                           
        tree.insert("hello", "tenant1");
        tree.insert("world", "tenant1");
        tree.insert("hello", "tenant2");
        tree.insert("help", "tenant2");

        let initial_sizes = tree.get_used_size_per_tenant();
        assert_eq!(initial_sizes.get("tenant1").unwrap(), &10);                     
        assert_eq!(initial_sizes.get("tenant2").unwrap(), &6);                 

                        
        tree.remove_tenant("tenant1");

        let final_sizes = tree.get_used_size_per_tenant();
        assert!(
            !final_sizes.contains_key("tenant1"),
            "tenant1 should be completely removed"
        );
        assert_eq!(
            final_sizes.get("tenant2").unwrap(),
            &6,
            "tenant2 should be unaffected"
        );

        assert_eq!(tree.prefix_match_tenant("hello", "tenant1"), "");
        assert_eq!(tree.prefix_match_tenant("world", "tenant1"), "");

        assert_eq!(tree.prefix_match_tenant("hello", "tenant2"), "hello");
        assert_eq!(tree.prefix_match_tenant("help", "tenant2"), "help");
    }

    #[test]
    fn test_complex_tenant_eviction() {
        let tree = Tree::new();

                                                                    
        tree.insert("apple", "tenant1");
        tree.insert("application", "tenant1");
        tree.insert("apple", "tenant2");
        tree.insert("appetite", "tenant2");
        tree.insert("banana", "tenant1");
        tree.insert("banana", "tenant2");
        tree.insert("ball", "tenant2");

        let initial_sizes = tree.get_used_size_per_tenant();
        println!("Initial sizes: {:?}", initial_sizes);
        tree.pretty_print();

                        
        tree.remove_tenant("tenant1");

        let final_sizes = tree.get_used_size_per_tenant();
        println!("Final sizes: {:?}", final_sizes);
        tree.pretty_print();

        assert!(
            !final_sizes.contains_key("tenant1"),
            "tenant1 should be completely removed"
        );

        assert_eq!(tree.prefix_match_tenant("apple", "tenant1"), "");
        assert_eq!(tree.prefix_match_tenant("application", "tenant1"), "");
        assert_eq!(tree.prefix_match_tenant("banana", "tenant1"), "");

        assert_eq!(tree.prefix_match_tenant("apple", "tenant2"), "apple");
        assert_eq!(tree.prefix_match_tenant("appetite", "tenant2"), "appetite");
        assert_eq!(tree.prefix_match_tenant("banana", "tenant2"), "banana");
        assert_eq!(tree.prefix_match_tenant("ball", "tenant2"), "ball");

        let tenant2_size = final_sizes.get("tenant2").unwrap();
        assert_eq!(tenant2_size, &(5 + 5 + 6 + 2));                                       
    }

                                                                

    #[test]
    fn test_empty_string_input() {
        let tree = Tree::new();

                              
        tree.insert("", "tenant1");

                             
        let (matched, tenant) = tree.prefix_match("");
        assert_eq!(matched, "");
        assert_eq!(tenant, "tenant1");

                                             
        tree.insert("hello", "tenant2");
        let (matched, tenant) = tree.prefix_match("");
        assert_eq!(matched, "");
        assert_eq!(tenant, "tenant1");
    }

    #[test]
    fn test_single_character_operations() {
        let tree = Tree::new();

                                   
        tree.insert("a", "tenant1");
        tree.insert("b", "tenant2");
        tree.insert("c", "tenant1");

        let (matched, tenant) = tree.prefix_match("a");
        assert_eq!(matched, "a");
        assert_eq!(tenant, "tenant1");

        let (matched, tenant) = tree.prefix_match("b");
        assert_eq!(matched, "b");
        assert_eq!(tenant, "tenant2");

                                                             
        let (matched, tenant) = tree.prefix_match("abc");
        assert_eq!(matched, "a");
        assert_eq!(tenant, "tenant1");
    }

    #[test]
    fn test_prefix_is_subset_of_existing() {
        let tree = Tree::new();

                                     
        tree.insert("application", "tenant1");

                                        
        tree.insert("app", "tenant2");

                                                         
        let (matched, tenant) = tree.prefix_match("app");
        assert_eq!(matched, "app");
        assert!(tenant == "tenant1" || tenant == "tenant2");

                              
        let (matched, tenant) = tree.prefix_match("application");
        assert_eq!(matched, "application");
        assert_eq!(tenant, "tenant1");

                                                                           
                                                                                  
        let (matched, _tenant) = tree.prefix_match("apple");
        assert_eq!(matched, "appl");
    }

    #[test]
    fn test_existing_is_prefix_of_new() {
        let tree = Tree::new();

                                      
        tree.insert("app", "tenant1");

                                                    
        tree.insert("application", "tenant2");

        let (matched, tenant) = tree.prefix_match("app");
        assert_eq!(matched, "app");
        assert!(tenant == "tenant1" || tenant == "tenant2");

        let (matched, tenant) = tree.prefix_match("application");
        assert_eq!(matched, "application");
        assert_eq!(tenant, "tenant2");

                                                                        
                                                                                  
        let (matched, _tenant) = tree.prefix_match("applesauce");
        assert_eq!(matched, "appl");
    }

                                                                               

    #[test]
    fn test_prefix_match_with_counts_accuracy() {
        let tree = Tree::new();

        tree.insert("hello world", "tenant1");

                      
        let result = tree.prefix_match_with_counts("hello world");
        assert_eq!(result.matched_char_count, 11);
        assert_eq!(result.input_char_count, 11);
        assert_eq!(&*result.tenant, "tenant1");

                        
        let result = tree.prefix_match_with_counts("hello");
        assert_eq!(result.matched_char_count, 5);
        assert_eq!(result.input_char_count, 5);

                         
        let result = tree.prefix_match_with_counts("hello world and more");
        assert_eq!(result.matched_char_count, 11);
        assert_eq!(result.input_char_count, 20);

                   
        let result = tree.prefix_match_with_counts("goodbye");
        assert_eq!(result.matched_char_count, 0);
        assert_eq!(result.input_char_count, 7);
    }

    #[test]
    fn test_prefix_match_with_counts_utf8() {
        let tree = Tree::new();

                                                 
        tree.insert("你好世界呀", "tenant1");

        let result = tree.prefix_match_with_counts("你好世界呀");
        assert_eq!(result.matched_char_count, 5);
        assert_eq!(result.input_char_count, 5);

        let result = tree.prefix_match_with_counts("你好");
        assert_eq!(result.matched_char_count, 2);
        assert_eq!(result.input_char_count, 2);

                                
        tree.insert("hello你好", "tenant2");
        let result = tree.prefix_match_with_counts("hello你好世界");
        assert_eq!(result.matched_char_count, 7);                       
        assert_eq!(result.input_char_count, 9);                         
    }

                                                                          

    #[test]
    fn test_split_at_first_character() {
        let tree = Tree::new();

                       
        tree.insert("abc", "tenant1");

                                                    
        tree.insert("aXX", "tenant2");

        let (matched, tenant) = tree.prefix_match("abc");
        assert_eq!(matched, "abc");
        assert_eq!(tenant, "tenant1");

        let (matched, tenant) = tree.prefix_match("aXX");
        assert_eq!(matched, "aXX");
        assert_eq!(tenant, "tenant2");

        let (matched, _) = tree.prefix_match("a");
        assert_eq!(matched, "a");
    }

    #[test]
    fn test_split_at_last_character() {
        let tree = Tree::new();

                        
        tree.insert("abcd", "tenant1");

                                                                     
        tree.insert("abcX", "tenant2");

        let (matched, tenant) = tree.prefix_match("abcd");
        assert_eq!(matched, "abcd");
        assert_eq!(tenant, "tenant1");

        let (matched, tenant) = tree.prefix_match("abcX");
        assert_eq!(matched, "abcX");
        assert_eq!(tenant, "tenant2");

        let (matched, _) = tree.prefix_match("abc");
        assert_eq!(matched, "abc");
    }

    #[test]
    fn test_multiple_splits_same_path() {
        let tree = Tree::new();

                                   
        tree.insert("abcdefgh", "tenant1");
        tree.insert("abcdef", "tenant2");
        tree.insert("abcd", "tenant3");
        tree.insert("ab", "tenant4");

                                
        assert_eq!(tree.prefix_match("abcdefgh").0, "abcdefgh");
        assert_eq!(tree.prefix_match("abcdef").0, "abcdef");
        assert_eq!(tree.prefix_match("abcd").0, "abcd");
        assert_eq!(tree.prefix_match("ab").0, "ab");
        assert_eq!(tree.prefix_match("a").0, "a");
    }

                                                                             

    #[test]
    fn test_high_contention_same_prefix() {
        let tree = Arc::new(Tree::new());
        let num_threads = 16;
        let ops_per_thread = 100;
        let mut handles = vec![];

                                                          
        for thread_id in 0..num_threads {
            let tree = Arc::clone(&tree);
            let handle = thread::spawn(move || {
                let tenant = format!("tenant{}", thread_id);
                for i in 0..ops_per_thread {
                    let text = format!("shared_prefix_{}", i);
                    tree.insert(&text, &tenant);

                                               
                    let (matched, _) = tree.prefix_match(&text);
                    assert!(
                        matched.starts_with("shared_prefix_"),
                        "Match should start with shared_prefix_"
                    );
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread panicked");
        }

                                          
        let sizes = tree.get_used_size_per_tenant();
        assert!(!sizes.is_empty(), "Tree should have entries");
    }

    #[test]
    fn test_rapid_insert_remove_cycles() {
        let tree = Arc::new(Tree::new());
        let num_cycles = 50;

        for cycle in 0..num_cycles {
            let tenant = format!("tenant{}", cycle % 5);

                                     
            for i in 0..10 {
                let text = format!("cycle{}entry{}", cycle, i);
                tree.insert(&text, &tenant);
            }

                                
            tree.remove_tenant(&tenant);

                                    
            let sizes = tree.get_used_size_per_tenant();
            assert!(
                !sizes.contains_key(&tenant),
                "Tenant {} should be removed after cycle {}",
                tenant,
                cycle
            );
        }
    }

                                                                              

    #[test]
    fn test_ascii_utf8_consistency() {
        let tree = Tree::new();

                       
        tree.insert("hello", "tenant1");

                                                       
        tree.insert("你好", "tenant2");

                       
        tree.insert("hello你好", "tenant3");

                                    
        assert_eq!(tree.prefix_match("hello").0, "hello");
        assert_eq!(tree.prefix_match("你好").0, "你好");
        assert_eq!(tree.prefix_match("hello你好").0, "hello你好");

                                   
        let result = tree.prefix_match_with_counts("hello");
        assert_eq!(result.matched_char_count, 5);
        assert_eq!(result.input_char_count, 5);

        let result = tree.prefix_match_with_counts("你好");
        assert_eq!(result.matched_char_count, 2);
        assert_eq!(result.input_char_count, 2);

        let result = tree.prefix_match_with_counts("hello你好");
        assert_eq!(result.matched_char_count, 7);
        assert_eq!(result.input_char_count, 7);
    }

    #[test]
    fn test_emoji_handling() {
        let tree = Tree::new();

                                     
        tree.insert("hello 👋", "tenant1");
        tree.insert("hello 👋🌍", "tenant2");

        let (matched, tenant) = tree.prefix_match("hello 👋");
        assert_eq!(matched, "hello 👋");
        assert_eq!(tenant, "tenant1");

        let (matched, tenant) = tree.prefix_match("hello 👋🌍");
        assert_eq!(matched, "hello 👋🌍");
        assert_eq!(tenant, "tenant2");

                                             
        let result = tree.prefix_match_with_counts("hello 👋");
        assert_eq!(result.matched_char_count, 7);
        assert_eq!(result.input_char_count, 7);                         
    }

                                                                    

    #[test]
    fn test_eviction_empty_tree() {
        let tree = Tree::new();

                                         
        tree.evict_tenant_by_size(100);

        let sizes = tree.get_used_size_per_tenant();
        assert!(sizes.is_empty());
    }

    #[test]
    fn test_eviction_zero_max_size() {
        let tree = Tree::new();

        tree.insert("hello", "tenant1");
        tree.insert("world", "tenant1");

                                                           
        tree.evict_tenant_by_size(0);

        let sizes = tree.get_used_size_per_tenant();
        assert!(
            sizes.is_empty() || sizes.values().all(|&v| v == 0),
            "All tenants should be evicted or have zero size"
        );
    }

    #[test]
    fn test_eviction_single_tenant_all_entries() {
        let tree = Tree::new();

                                                
        for i in 0..100 {
            let text = format!("entry{:03}", i);
            tree.insert(&text, "tenant1");
        }

        let initial_size = *tree.get_used_size_per_tenant().get("tenant1").unwrap();
        assert!(initial_size > 50, "Should have significant size");

                              
        tree.evict_tenant_by_size(50);

        let final_size = *tree.get_used_size_per_tenant().get("tenant1").unwrap_or(&0);
        assert!(
            final_size <= 50,
            "Size {} should be <= 50 after eviction",
            final_size
        );
    }

                                                                        

    #[test]
    fn test_last_tenant_cache_update() {
        let tree = Tree::new();

                             
        tree.insert("hello", "tenant1");

                                            
        let (_, tenant) = tree.prefix_match("hello");
        assert_eq!(tenant, "tenant1");

                                          
        tree.insert("hello", "tenant2");

                                                               
        let (matched, _) = tree.prefix_match("hello");
        assert_eq!(matched, "hello");
    }

    #[test]
    fn test_stale_cache_after_tenant_removal() {
        let tree = Tree::new();

        tree.insert("hello", "tenant1");
        tree.insert("hello", "tenant2");

                                   
        let _ = tree.prefix_match("hello");

                         
        tree.remove_tenant("tenant1");

                                         
        let (matched, tenant) = tree.prefix_match("hello");
        assert_eq!(matched, "hello");
        assert_eq!(tenant, "tenant2");
    }

                                                                               

    #[test]
    fn test_char_count_consistency_after_operations() {
        let tree = Tree::new();

                                       
        let verify_consistency = |tree: &Tree| {
            let maintained = get_maintained_counts(tree);
            let computed = tree.get_used_size_per_tenant();
            assert_eq!(
                maintained, computed,
                "Maintained counts should match computed counts"
            );
        };

                       
        for i in 0..50 {
            tree.insert(&format!("prefix{}", i), "tenant1");
            tree.insert(&format!("other{}", i), "tenant2");
        }
        verify_consistency(&tree);

                              
        for i in 0..25 {
            tree.insert(&format!("prefix{}", i), "tenant2");
        }
        verify_consistency(&tree);

                   
        tree.evict_tenant_by_size(100);
        verify_consistency(&tree);

                         
        tree.remove_tenant("tenant1");
        verify_consistency(&tree);
    }

    #[test]
    fn test_tree_structure_integrity_after_stress() {
        let tree = Arc::new(Tree::new());
        let num_threads = 8;
        let mut handles = vec![];

        for thread_id in 0..num_threads {
            let tree = Arc::clone(&tree);
            let handle = thread::spawn(move || {
                let mut rng = rand::rng();
                let tenant = format!("tenant{}", thread_id);

                for _ in 0..200 {
                    let op: u8 = rng.random_range(0..10);
                    let key = format!("key{}", rng.random_range(0..50));

                    match op {
                        0..=6 => {
                                           
                            tree.insert(&key, &tenant);
                        }
                        7..=8 => {
                                          
                            let _ = tree.prefix_match(&key);
                        }
                        _ => {
                                                      
                            let _ = tree.prefix_match_with_counts(&key);
                        }
                    }
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread panicked during stress test");
        }

                                          
        let sizes = tree.get_used_size_per_tenant();
        for (tenant, size) in sizes.iter() {
            assert!(*size > 0, "Tenant {} should have positive size", tenant);
        }

                                        
        let maintained = get_maintained_counts(&tree);
        let computed = tree.get_used_size_per_tenant();
        assert_eq!(
            maintained, computed,
            "Counts should be consistent after stress test"
        );
    }

                                                                         

    #[test]
    fn test_very_long_strings() {
        let tree = Tree::new();

                                           
        let long_string: String = (0..10000)
            .map(|i| ((i % 26) as u8 + b'a') as char)
            .collect();

        tree.insert(&long_string, "tenant1");

        let (matched, tenant) = tree.prefix_match(&long_string);
        assert_eq!(matched.len(), long_string.len());
        assert_eq!(tenant, "tenant1");

                                       
        let partial = &long_string[..5000];
        let (matched, _) = tree.prefix_match(partial);
        assert_eq!(matched, partial);
    }

    #[test]
    fn test_many_tenants_same_path() {
        let tree = Tree::new();

                                             
        for i in 0..100 {
            tree.insert("shared_path", &format!("tenant{}", i));
        }

                                          
        let (matched, _) = tree.prefix_match("shared_path");
        assert_eq!(matched, "shared_path");

                                         
        let sizes = tree.get_used_size_per_tenant();
        assert_eq!(sizes.len(), 100, "Should have 100 tenants");
    }

    #[test]
    fn test_special_characters() {
        let tree = Tree::new();

                                     
        let test_cases = vec![
            ("hello\nworld", "tenant1"),                
            ("hello\tworld", "tenant2"),            
            ("hello\0world", "tenant3"),                  
            ("hello\u{A0}world", "tenant4"),                       
            ("path/to/file", "tenant5"),                
            ("query?param=value", "tenant6"),            
        ];

        for (text, tenant) in &test_cases {
            tree.insert(text, tenant);
        }

        for (text, tenant) in &test_cases {
            let (matched, matched_tenant) = tree.prefix_match(text);
            assert_eq!(matched, *text, "Failed for: {:?}", text);
            assert_eq!(matched_tenant, *tenant);
        }
    }
}

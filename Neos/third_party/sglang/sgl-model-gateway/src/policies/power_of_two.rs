//! Power-of-two choices load balancing policy

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use async_trait::async_trait;
use rand::Rng;
use tracing::debug;

use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
use crate::core::Worker;

/// Power-of-two choices policy
///
/// Randomly selects two workers and routes to the one with lower load.
/// This provides good load distribution with minimal coordination overhead.
#[derive(Debug)]
pub struct PowerOfTwoPolicy {
    /// Cached load information from external monitoring
    cached_loads: RwLock<HashMap<String, isize>>,
}

impl PowerOfTwoPolicy {
    pub fn new() -> Self {
        Self {
            cached_loads: RwLock::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl LoadBalancingPolicy for PowerOfTwoPolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        _info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        if healthy_indices.len() == 1 {
            return Some(healthy_indices[0]);
        }

                                                                                          
        let mut rng = rand::rng();
        let idx1 = rng.random_range(0..healthy_indices.len());
                                                                                                     
        let idx2 =
            (idx1 + 1 + rng.random_range(0..healthy_indices.len() - 1)) % healthy_indices.len();

        let worker_idx1 = healthy_indices[idx1];
        let worker_idx2 = healthy_indices[idx2];
        let worker1 = &workers[worker_idx1];
        let worker2 = &workers[worker_idx2];

                                     
        let loads_guard = self.cached_loads.read().ok();

                                                                
        let load1_tokens = loads_guard
            .as_ref()
            .and_then(|m| m.get(worker1.url()).copied());
        let load2_tokens = loads_guard
            .as_ref()
            .and_then(|m| m.get(worker2.url()).copied());

                                                                         
                                                                     
        let (load1, load2) = match (load1_tokens, load2_tokens) {
            (Some(t1), Some(t2)) => {
                                                        
                (t1, t2)
            }
            _ => {
                                                         
                                                             
                (worker1.load() as isize, worker2.load() as isize)
            }
        };

                                        
        let selected_idx = if load1 <= load2 {
            worker_idx1
        } else {
            worker_idx2
        };

        debug!(
            "Power-of-two selection: {}={} vs {}={} -> selected {}",
            worker1.url(),
            load1,
            worker2.url(),
            load2,
            workers[selected_idx].url()
        );

                                      
        workers[selected_idx].increment_processed();

        Some(selected_idx)
    }

    fn name(&self) -> &'static str {
        "power_of_two"
    }

    fn update_loads(&self, loads: &HashMap<String, isize>) {
        if let Ok(mut cached) = self.cached_loads.write() {
            *cached = loads.clone();
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Default for PowerOfTwoPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    #[tokio::test]
    async fn test_power_of_two_selection() {
        let policy = PowerOfTwoPolicy::new();
        let worker1 = BasicWorkerBuilder::new("http://w1:8000")
            .worker_type(WorkerType::Regular)
            .build();
        let worker2 = BasicWorkerBuilder::new("http://w2:8000")
            .worker_type(WorkerType::Regular)
            .build();
        let worker3 = BasicWorkerBuilder::new("http://w3:8000")
            .worker_type(WorkerType::Regular)
            .build();

                              
        for _ in 0..10 {
            worker1.increment_load();
        }
        for _ in 0..5 {
            worker2.increment_load();
        }
                             

        let workers: Vec<Arc<dyn Worker>> =
            vec![Arc::new(worker1), Arc::new(worker2), Arc::new(worker3)];

                                  
        let mut selected_counts = [0; 3];
        let info = SelectWorkerInfo::default();
        for _ in 0..100 {
            if let Some(idx) = policy.select_worker(&workers, &info).await {
                selected_counts[idx] += 1;
            }
        }

                                                                          
        assert!(selected_counts[2] > selected_counts[1]);
        assert!(selected_counts[1] > selected_counts[0]);
    }

    #[tokio::test]
    async fn test_power_of_two_with_cached_loads() {
        let policy = PowerOfTwoPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
        ];

                              
        let mut loads = HashMap::new();
        loads.insert("http://w1:8000".to_string(), 100);
        loads.insert("http://w2:8000".to_string(), 10);
        policy.update_loads(&loads);

                                                       
        let mut w2_selected = 0;
        let info = SelectWorkerInfo::default();
        for _ in 0..50 {
            if let Some(idx) = policy.select_worker(&workers, &info).await {
                if idx == 1 {
                    w2_selected += 1;
                }
            }
        }

                                                              
        assert!(w2_selected > 35);                               
    }

    #[tokio::test]
    async fn test_power_of_two_single_worker() {
        let policy = PowerOfTwoPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        )];

                                                      
        assert_eq!(
            policy
                .select_worker(&workers, &SelectWorkerInfo::default())
                .await,
            Some(0)
        );
    }

    #[tokio::test]
    async fn test_reproduce_incompatible_metric_bug() {
        use std::{collections::HashMap, sync::Arc};

        use crate::core::{BasicWorkerBuilder, WorkerType};

                              
        let policy = PowerOfTwoPolicy::new();

                                                                               
        let worker_a = BasicWorkerBuilder::new("http://worker_a:8000")
            .worker_type(WorkerType::Regular)
            .build();

                                                                    
        let worker_b = BasicWorkerBuilder::new("http://worker_b:8000")
            .worker_type(WorkerType::Regular)
            .build();

                                                                          
        for _ in 0..5 {
            worker_b.increment_load();
        }

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(worker_a), Arc::new(worker_b)];

                                          
                                                                                         
        let mut loads = HashMap::new();
        loads.insert("http://worker_a:8000".to_string(), 50_000);                   
        policy.update_loads(&loads);

                           
        let selected_idx = policy
            .select_worker(&workers, &SelectWorkerInfo::default())
            .await
            .expect("Should select a worker");

                            
                 
                                                                          
                                                               
                                                
                                             

        if selected_idx == 0 {
            println!("Bug Fixed: System correctly fell back to request counts and selected idle Worker A.");
        } else {
            println!(
                "Bug PERSISTS: Selected Worker B (Load: 5 reqs) over Worker A (Load: 50k tokens)"
            );
        }

                                                                  
        assert_eq!(
            selected_idx, 0,
            "The policy failed to handle incompatible metrics. Should select idle Worker A."
        );
    }
    #[tokio::test]
    async fn test_power_of_two_edge_cases() {
        use std::{collections::HashMap, sync::Arc};

        use crate::core::{BasicWorkerBuilder, WorkerType};

        let policy = PowerOfTwoPolicy::new();

                                                               
        let create_worker = |url: &str, reqs: usize| {
            let w = BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .build();
            for _ in 0..reqs {
                w.increment_load();
            }
            Arc::new(w)
        };

                                                         
                                                                                         
                                                                                       
                                                                                            
        let w_a = create_worker("http://a:8000", 10);
        let w_b = create_worker("http://b:8000", 2);
        let workers_1: Vec<Arc<dyn Worker>> = vec![w_a.clone(), w_b.clone()];

        let mut loads_1 = HashMap::new();
        loads_1.insert("http://a:8000".to_string(), 1_000);
        loads_1.insert("http://b:8000".to_string(), 100_000);
        policy.update_loads(&loads_1);

        let idx_1 = policy
            .select_worker(&workers_1, &SelectWorkerInfo::default())
            .await
            .unwrap();
        assert_eq!(
            idx_1, 0,
            "Happy Path Failed: Should select Worker A (fewer tokens) despite higher request count"
        );

                                                                                 
                                                       
                                               
                                                                             
        let w_c = create_worker("http://c:8000", 10);
        let w_d = create_worker("http://d:8000", 2);
        let workers_2: Vec<Arc<dyn Worker>> = vec![w_c.clone(), w_d.clone()];

        let mut loads_2 = HashMap::new();
        loads_2.insert("http://c:8000".to_string(), 1_000);
                                   
        policy.update_loads(&loads_2);

        let idx_2 = policy
            .select_worker(&workers_2, &SelectWorkerInfo::default())
            .await
            .unwrap();
        assert_eq!(idx_2, 1, "Partial Fail 1 Failed: Should fallback to requests and select Worker B (fewer requests)");

                                                                                 
                                               
                                                       
                                                                             
        let w_e = create_worker("http://e:8000", 2);
        let w_f = create_worker("http://f:8000", 10);
        let workers_3: Vec<Arc<dyn Worker>> = vec![w_e.clone(), w_f.clone()];

        let mut loads_3 = HashMap::new();
                                   
        loads_3.insert("http://f:8000".to_string(), 1_000);
        policy.update_loads(&loads_3);

        let idx_3 = policy
            .select_worker(&workers_3, &SelectWorkerInfo::default())
            .await
            .unwrap();
        assert_eq!(idx_3, 0, "Partial Fail 2 Failed: Should fallback to requests and select Worker A (fewer requests)");

                                                   
                               
                               
                                                  
        let w_g = create_worker("http://g:8000", 5);
        let w_h = create_worker("http://h:8000", 3);
        let workers_4: Vec<Arc<dyn Worker>> = vec![w_g.clone(), w_h.clone()];

        let loads_4 = HashMap::new();
        policy.update_loads(&loads_4);

        let idx_4 = policy
            .select_worker(&workers_4, &SelectWorkerInfo::default())
            .await
            .unwrap();
        assert_eq!(
            idx_4, 1,
            "Total Fail Failed: Should select Worker B based on request count"
        );

        println!("All edge case tests passed successfully.");
    }
}

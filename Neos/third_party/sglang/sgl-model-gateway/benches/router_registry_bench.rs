use std::{collections::HashMap, sync::Arc};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use smg::core::{BasicWorkerBuilder, CircuitBreakerConfig, WorkerRegistry, WorkerType};

                              
fn setup_registry(count: usize) -> Arc<WorkerRegistry> {
    let registry = Arc::new(WorkerRegistry::new());

    for i in 0..count {
        let mut labels = HashMap::new();
        labels.insert("model_id".to_string(), "benchmark-model".to_string());

        let worker_type = if i % 2 == 0 {
            WorkerType::Regular
        } else {
            WorkerType::Decode
        };

        let worker = BasicWorkerBuilder::new(format!("http://worker-{}:8000", i))
            .worker_type(worker_type)
            .labels(labels)
            .circuit_breaker_config(CircuitBreakerConfig::default())
            .build();

        registry.register(Arc::from(worker));
    }
    registry
}

fn bench_optimizations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Registry Optimizations");

                                                      
    let size = 5000;
    let registry = setup_registry(size);

                                                             
    group.bench_function(BenchmarkId::new("Old: get_all()", size), |b| {
        b.iter(|| {
            black_box(registry.get_all());
        });
    });

                                                           
    group.bench_function(
        BenchmarkId::new("New: get_worker_distribution()", size),
        |b| {
            b.iter(|| {
                black_box(registry.get_worker_distribution());
            });
        },
    );

    group.finish();
}

criterion_group!(benches, bench_optimizations);
criterion_main!(benches);

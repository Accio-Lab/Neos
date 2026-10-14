//! Circuit breaker integration tests
//!
//! Tests for circuit breaker behavior: opening, half-open state, recovery, and isolation.

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use smg::config::{CircuitBreakerConfig, RetryConfig, RouterConfig};
use tower::ServiceExt;

use crate::common::{AppTestContext, TestRouterConfig, TestWorkerConfig};

#[cfg(test)]
mod circuit_breaker_tests {
    use super::*;

    /// Test that circuit breaker opens after consecutive failures
    #[tokio::test]
    async fn test_circuit_breaker_opens_after_failures() {
        let config = TestRouterConfig::round_robin_with_circuit_breaker(
            3200,
            CircuitBreakerConfig {
                failure_threshold: 3,
                success_threshold: 2,
                timeout_duration_secs: 2,
                window_duration_secs: 10,
            },
        );

        let ctx = AppTestContext::new_with_config(
            config,
            vec![TestWorkerConfig::flaky(19100, 1.0)],               
        )
        .await;

        let app = ctx.create_app().await;

                                                                                         
        let mut saw_500 = false;
        let mut saw_503 = false;

        for _ in 0..10 {
            let payload = json!({
                "text": "Test circuit breaker",
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            match resp.status() {
                StatusCode::INTERNAL_SERVER_ERROR => saw_500 = true,
                StatusCode::SERVICE_UNAVAILABLE => saw_503 = true,
                _ => {}
            }
        }

                                                                                  
        assert!(
            saw_500 || saw_503,
            "Should see either 500 (worker error) or 503 (circuit breaker open)"
        );

        ctx.shutdown().await;
    }

    /// Test circuit breaker with disabled flag
    #[tokio::test]
    async fn test_circuit_breaker_disabled() {
        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .round_robin_policy()
            .host("127.0.0.1")
            .port(3201)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .disable_circuit_breaker()
            .disable_retries()
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![TestWorkerConfig::flaky(19101, 1.0)],               
        )
        .await;

        let app = ctx.create_app().await;

                                                                           
        for _ in 0..5 {
            let payload = json!({
                "text": "Test disabled CB",
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
                                                                       
            assert_eq!(
                resp.status(),
                StatusCode::INTERNAL_SERVER_ERROR,
                "Should get 500 with CB disabled, not CB-related 503"
            );
        }
        ctx.shutdown().await;
    }

    /// Test circuit breaker per-worker isolation
    #[tokio::test]
    async fn test_circuit_breaker_per_worker_isolation() {
        let config = TestRouterConfig::round_robin_with_circuit_breaker(
            3202,
            CircuitBreakerConfig {
                failure_threshold: 2,
                success_threshold: 1,
                timeout_duration_secs: 2,
                window_duration_secs: 10,
            },
        );

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::flaky(19102, 1.0),               
                TestWorkerConfig::healthy(19103),                     
            ],
        )
        .await;

        let app = ctx.create_app().await;

                                                                   
                                                               
        let mut success_count = 0;
        for _ in 0..20 {
            let payload = json!({
                "text": "Test isolation",
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            if resp.status() == StatusCode::OK {
                success_count += 1;
            }
        }

                                                                                      
                                                
        assert!(
            success_count > 0,
            "Should have some successful requests from healthy worker"
        );

        ctx.shutdown().await;
    }

    /// Test circuit breaker with retries enabled
    #[tokio::test]
    async fn test_circuit_breaker_with_retries() {
        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .round_robin_policy()
            .host("127.0.0.1")
            .port(3203)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .retry_config(RetryConfig {
                max_retries: 3,
                initial_backoff_ms: 10,
                max_backoff_ms: 50,
                ..Default::default()
            })
            .circuit_breaker_config(CircuitBreakerConfig {
                failure_threshold: 2,
                success_threshold: 1,
                timeout_duration_secs: 2,
                window_duration_secs: 10,
            })
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::flaky(19104, 1.0),               
                TestWorkerConfig::healthy(19105),                     
            ],
        )
        .await;

        let app = ctx.create_app().await;

                                                                              
        let payload = json!({
            "text": "Test with retries",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Request should succeed via retry on healthy worker"
        );

        ctx.shutdown().await;
    }
}

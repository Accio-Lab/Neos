use std::sync::Arc;

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use reqwest::Client;
use serde_json::json;
use smg::{
    app_context::AppContext,
    config::{RouterConfig, RoutingMode},
    routers::{RouterFactory, RouterTrait},
};
use tower::ServiceExt;

use crate::common::mock_worker::{MockWorker, MockWorkerConfig};

/// Test context that manages mock workers and app
struct ParserTestContext {
    workers: Vec<MockWorker>,
    router: Arc<dyn RouterTrait>,
    _client: Client,
    _config: RouterConfig,
    app_context: Arc<AppContext>,
}

impl ParserTestContext {
    async fn new(worker_configs: Vec<MockWorkerConfig>) -> Self {
                                                           
        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .random_policy()
            .host("127.0.0.1")
            .port(3003)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(1)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        Self::new_with_config(config, worker_configs).await
    }

    async fn new_with_config(
        mut config: RouterConfig,
        worker_configs: Vec<MockWorkerConfig>,
    ) -> Self {
        let mut workers = Vec::new();
        let mut worker_urls = Vec::new();

                                    
        for worker_config in worker_configs {
            let mut worker = MockWorker::new(worker_config);
            let url = worker.start().await.unwrap();
            worker_urls.push(url);
            workers.push(worker);
        }

        if !workers.is_empty() {
            tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        }

                                                            
        match &mut config.mode {
            RoutingMode::Regular {
                worker_urls: ref mut urls,
            } => {
                if urls.is_empty() {
                    *urls = worker_urls.clone();
                }
            }
            RoutingMode::OpenAI {
                worker_urls: ref mut urls,
            } => {
                if urls.is_empty() {
                    *urls = worker_urls.clone();
                }
            }
            _ => {}                                        
        }

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.request_timeout_secs))
            .build()
            .unwrap();

                                                               
        let app_context = crate::common::create_test_context_with_parsers(config.clone()).await;

                        
        let router = RouterFactory::create_router(&app_context).await.unwrap();
        let router = Arc::from(router);

        Self {
            workers,
            router,
            _client: client,
            _config: config,
            app_context,
        }
    }

    async fn create_app(&self) -> axum::Router {
        crate::common::test_app::create_test_app_with_context(
            Arc::clone(&self.router),
            Arc::clone(&self.app_context),
        )
    }

    async fn shutdown(mut self) {
        for worker in &mut self.workers {
            worker.stop().await;
        }
    }
}

#[cfg(test)]
mod parse_function_call_tests {
    use super::*;

    #[tokio::test]
    async fn test_parse_function_call_success() {
        let ctx = ParserTestContext::new(vec![]).await;
        let app = ctx.create_app().await;

        let payload = json!({
            "text": r#"I need to call the weather function <tool_call>{"function_name": "get_weather", "parameters": {"location": "Beijing"}}</tool_call>"#,
            "tool_call_parser": "json",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather information",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The location"
                                }
                            },
                            "required": ["location"]
                        }
                    }
                }
            ]
        });

        let req = Request::builder()
            .method("POST")
            .uri("/parse/function_call")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();

                                                               
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Expected OK (200), got {}",
            resp.status()
        );

                                              
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body_json["success"], true);
        assert!(body_json["tool_calls"].is_array());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_parse_function_call_invalid_parser() {
        let ctx = ParserTestContext::new(vec![]).await;
        let app = ctx.create_app().await;

        let payload = json!({
            "text": "some text",
            "tool_call_parser": "nonexistent_parser",
            "tools": []
        });

        let req = Request::builder()
            .method("POST")
            .uri("/parse/function_call")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();

                                               
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "Expected BAD_REQUEST (400), got {}",
            resp.status()
        );

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(body_json["success"], false);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_parse_function_call_missing_fields() {
        let ctx = ParserTestContext::new(vec![]).await;
        let app = ctx.create_app().await;

                               
        let payload = json!({
            "tool_call_parser": "json",
            "tools": []
        });

        let req = Request::builder()
            .method("POST")
            .uri("/parse/function_call")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();

        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);

                                           
        let payload = json!({
            "text": "some text",
            "tools": []
        });

        let req = Request::builder()
            .method("POST")
            .uri("/parse/function_call")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_parse_function_call_empty_text() {
        let ctx = ParserTestContext::new(vec![]).await;
        let app = ctx.create_app().await;

        let payload = json!({
            "text": "",
            "tool_call_parser": "json",
            "tools": []
        });

        let req = Request::builder()
            .method("POST")
            .uri("/parse/function_call")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();

                                                                  
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Expected OK (200), got {}",
            resp.status()
        );

        ctx.shutdown().await;
    }
}

#[cfg(test)]
mod separate_reasoning_tests {
    use super::*;

    #[tokio::test]
    async fn test_separate_reasoning_success() {
        let ctx = ParserTestContext::new(vec![]).await;
        let app = ctx.create_app().await;

        let payload = json!({
            "text": "<think>Let me think about this problem. The user is asking for help.</think>Sure, I can help you with that.",
            "reasoning_parser": "step3"
        });

        let req = Request::builder()
            .method("POST")
            .uri("/parse/reasoning")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();

                                                            
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Expected OK (200), got {}",
            resp.status()
        );

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();

                                   
        assert_eq!(body_json["success"], true);
        assert!(body_json.get("normal_text").is_some());
        assert!(body_json.get("reasoning_text").is_some());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_separate_reasoning_invalid_parser() {
        let ctx = ParserTestContext::new(vec![]).await;
        let app = ctx.create_app().await;

        let payload = json!({
            "text": "some text",
            "reasoning_parser": "invalid_parser_type"
        });

        let req = Request::builder()
            .method("POST")
            .uri("/parse/reasoning")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();

                                               
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "Expected BAD_REQUEST (400), got {}",
            resp.status()
        );

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(body_json["success"], false);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_separate_reasoning_missing_fields() {
        let ctx = ParserTestContext::new(vec![]).await;
        let app = ctx.create_app().await;

                               
        let payload = json!({
            "reasoning_parser": "step3"
        });

        let req = Request::builder()
            .method("POST")
            .uri("/parse/reasoning")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();

        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);

                                           
        let payload = json!({
            "text": "some text"
        });

        let req = Request::builder()
            .method("POST")
            .uri("/parse/reasoning")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_separate_reasoning_empty_text() {
        let ctx = ParserTestContext::new(vec![]).await;
        let app = ctx.create_app().await;

        let payload = json!({
            "text": "",
            "reasoning_parser": "step3"
        });

        let req = Request::builder()
            .method("POST")
            .uri("/parse/reasoning")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();

                                                     
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Expected OK (200), got {}",
            resp.status()
        );

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_separate_reasoning_without_reasoning_tags() {
        let ctx = ParserTestContext::new(vec![]).await;
        let app = ctx.create_app().await;

        let payload = json!({
            "text": "Just a normal text without any reasoning tags",
            "reasoning_parser": "step3"
        });

        let req = Request::builder()
            .method("POST")
            .uri("/parse/reasoning")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();

                                                             
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Expected OK (200), got {}",
            resp.status()
        );

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(body_json["success"], true);
                                                                                                      
                                                                                                  
        assert!(body_json.get("normal_text").is_some());
        assert!(body_json.get("reasoning_text").is_some());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_separate_reasoning_multiple_reasoning_blocks() {
        let ctx = ParserTestContext::new(vec![]).await;
        let app = ctx.create_app().await;

                                                            
        let payload = json!({
            "text": "<think>First thought</think>Text 1<think>Second thought</think>Text 2",
            "reasoning_parser": "step3"
        });

        let req = Request::builder()
            .method("POST")
            .uri("/parse/reasoning")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();

                                                   
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Expected OK (200), got {}",
            resp.status()
        );

        ctx.shutdown().await;
    }
}

#[cfg(test)]
mod api_routing_tests {
    use super::*;

    #[tokio::test]
    async fn test_admin_routes_accessible() {
        let ctx = ParserTestContext::new(vec![]).await;
        let app = ctx.create_app().await;

                                                                                                     
        let payload = json!({
            "text": "test",
            "tool_call_parser": "json",
            "tools": []
        });

        let req = Request::builder()
            .method("POST")
            .uri("/parse/function_call")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();

                            
        assert_ne!(resp.status(), StatusCode::NOT_FOUND);

        let payload = json!({
            "text": "test",
            "reasoning_parser": "step3"
        });

        let req = Request::builder()
            .method("POST")
            .uri("/parse/reasoning")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();

                            
        assert_ne!(resp.status(), StatusCode::NOT_FOUND);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_endpoints_only_accept_post() {
        let ctx = ParserTestContext::new(vec![]).await;
        let app = ctx.create_app().await;

                                                  
        let req = Request::builder()
            .method("GET")
            .uri("/parse/function_call")
            .body(Body::empty())
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();

                                                       
        assert!(
            resp.status() == StatusCode::METHOD_NOT_ALLOWED
                || resp.status() == StatusCode::NOT_FOUND,
            "Expected METHOD_NOT_ALLOWED (405) or NOT_FOUND (404), got {}",
            resp.status()
        );

                                              
        let req = Request::builder()
            .method("GET")
            .uri("/parse/reasoning")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();

                                
        assert!(
            resp.status() == StatusCode::METHOD_NOT_ALLOWED
                || resp.status() == StatusCode::NOT_FOUND,
            "Expected METHOD_NOT_ALLOWED (405) or NOT_FOUND (404), got {}",
            resp.status()
        );

        ctx.shutdown().await;
    }
}

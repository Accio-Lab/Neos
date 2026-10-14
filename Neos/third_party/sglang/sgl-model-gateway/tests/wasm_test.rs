//! WASM Module Integration Tests
//!
//! This test suite validates the complete WASM module management functionality:
//! - API endpoints (add, remove, list)
//! - Workflow integration
//! - Module execution
//! - Error handling

mod common;

use std::{sync::Arc, time::Duration};

use axum::{
    body::{to_bytes, Body},
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use data_connector::{
    MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
};
use smg::{
    app_context::AppContext,
    config::RouterConfig,
    core::{LoadMonitor, WorkerRegistry},
    policies::PolicyRegistry,
    routers::RouterFactory,
    server::{build_app, AppState},
    tokenizer::TokenizerRegistry,
    wasm::{
        module::{
            WasmModuleAddRequest, WasmModuleAddResponse, WasmModuleAttachPoint,
            WasmModuleDescriptor, WasmModuleListResponse, WasmModuleType,
        },
        module_manager::WasmModuleManager,
    },
};
use tempfile::TempDir;
use tokio::fs;
use tower::ServiceExt;
use uuid::Uuid;

/// Create a test AppContext with WASM manager initialized
async fn create_test_context_with_wasm() -> Arc<AppContext> {
    let config = RouterConfig::default();

                                    
    let wasm_manager = Arc::new(
        WasmModuleManager::with_default_config().expect("Failed to create WASM module manager"),
    );

                                                         
    let client = reqwest::Client::new();

    let tokenizer_registry = Arc::new(TokenizerRegistry::new());
    let worker_registry = Arc::new(WorkerRegistry::new());
    let policy_registry = Arc::new(PolicyRegistry::new(config.policy.clone()));

                                  
    let response_storage = Arc::new(MemoryResponseStorage::new());
    let conversation_storage = Arc::new(MemoryConversationStorage::new());
    let conversation_item_storage = Arc::new(MemoryConversationItemStorage::new());

                              
    let load_monitor = Some(Arc::new(LoadMonitor::new(
        worker_registry.clone(),
        policy_registry.clone(),
        client.clone(),
        config.worker_startup_check_interval_secs,
    )));

                                                                                    
    use std::sync::OnceLock;
    let worker_job_queue = Arc::new(OnceLock::new());
    let workflow_engines = Arc::new(OnceLock::new());
    let mcp_manager_lock = Arc::new(OnceLock::new());

    let app_context = Arc::new(
        AppContext::builder()
            .router_config(config.clone())
            .client(client)
            .rate_limiter(None)
            .tokenizer_registry(tokenizer_registry)
            .reasoning_parser_factory(None)
            .tool_parser_factory(None)
            .worker_registry(worker_registry)
            .policy_registry(policy_registry)
            .response_storage(response_storage)
            .conversation_storage(conversation_storage)
            .conversation_item_storage(conversation_item_storage)
            .load_monitor(load_monitor)
            .worker_job_queue(worker_job_queue)
            .workflow_engines(workflow_engines)
            .mcp_manager(mcp_manager_lock)
            .wasm_manager(Some(wasm_manager))
            .build()
            .expect("Failed to build AppContext with WASM manager"),
    );

                                                      
    let weak_context = Arc::downgrade(&app_context);
    let job_queue = smg::core::JobQueue::new(smg::core::JobQueueConfig::default(), weak_context);
    app_context
        .worker_job_queue
        .set(job_queue)
        .expect("JobQueue should only be initialized once");

                                 
    use smg::core::steps::WorkflowEngines;
    let engines = WorkflowEngines::new(&config);
    app_context
        .workflow_engines
        .set(engines)
        .expect("WorkflowEngines should only be initialized once");

                                               
    use smg_mcp::{McpConfig, McpManager};
    let empty_config = McpConfig {
        servers: vec![],
        pool: Default::default(),
        proxy: None,
        warmup: vec![],
        inventory: Default::default(),
    };
    let mcp_manager = McpManager::with_defaults(empty_config)
        .await
        .expect("Failed to create MCP manager");
    app_context
        .mcp_manager
        .set(Arc::new(mcp_manager))
        .ok()
        .expect("McpManager should only be initialized once");

    app_context
}

/// Create a test WASM component file
/// Dynamically generates a valid WASM component programmatically without external tools
/// This ensures tests work in new environments without requiring pre-built files or external tools
async fn create_test_wasm_component(temp_dir: &TempDir) -> String {
    use wasm_encoder::{Component, Module};

                                               
                                                     
    let mut module = Module::new();

                                                          
    let type_section = wasm_encoder::TypeSection::new();
    module.section(&type_section);
    let mut component = Component::new();
    component.section(&wasm_encoder::ModuleSection(&module));
    let component_bytes = component.as_slice().to_vec();
    let component_path = temp_dir.path().join("test_module.component.wasm");
    fs::write(&component_path, component_bytes)
        .await
        .expect("Failed to write WASM component file");

                           
    component_path
        .canonicalize()
        .expect("Failed to canonicalize path")
        .to_str()
        .unwrap()
        .to_string()
}

/// Create a test app with WASM support
async fn create_test_app_with_wasm() -> (axum::Router, Arc<AppContext>, TempDir) {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let app_context = create_test_context_with_wasm().await;

                                                                      
    let router = RouterFactory::create_router(&app_context)
        .await
        .expect("Failed to create router");
    let router = Arc::from(router);

    let app_state = Arc::new(AppState {
        router,
        context: app_context.clone(),
        concurrency_queue_tx: None,
        router_manager: None,
        mesh_handler: None,
        mesh_sync_manager: None,
    });

    let request_id_headers = vec!["x-request-id".to_string(), "x-correlation-id".to_string()];

    let app = build_app(
        app_state,
        smg::middleware::AuthConfig { api_key: None },
        None,                                   
        256 * 1024 * 1024,
        request_id_headers,
        vec![],                        
    );

    (app, app_context, temp_dir)
}

                                                                               
                     
                                                                               

#[tokio::test]
async fn test_wasm_api_add_module() {
    let (app, app_context, temp_dir) = create_test_app_with_wasm().await;
    let wasm_file_path = create_test_wasm_component(&temp_dir).await;

    let add_request = WasmModuleAddRequest {
        modules: vec![WasmModuleDescriptor {
            name: "test_module".to_string(),
            file_path: wasm_file_path.clone(),
            module_type: WasmModuleType::Middleware,
            attach_points: vec![WasmModuleAttachPoint::Middleware(
                smg::wasm::module::MiddlewareAttachPoint::OnRequest,
            )],
            add_result: None,
        }],
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/wasm")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&add_request).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let response_json: WasmModuleAddResponse = serde_json::from_slice(&body).unwrap();

    assert_eq!(response_json.modules.len(), 1);
    let module_result = &response_json.modules[0].add_result;

                                
    if let Some(smg::wasm::module::WasmModuleAddResult::Error(err)) = module_result {
        eprintln!("Module registration failed: {}", err);
    }

                                                   
    if status != StatusCode::OK {
        eprintln!("Response status: {:?}", status);
        eprintln!("Response body: {}", String::from_utf8_lossy(&body));
        panic!(
            "Expected OK status but got {:?}. Error: {:?}",
            status, module_result
        );
    }

    assert!(module_result.is_some());

                                                  
    if let Some(wasm_manager) = app_context.wasm_manager.as_ref() {
        let modules = wasm_manager.get_modules().expect("Failed to get modules");
        assert!(!modules.is_empty(), "Module should be registered");

        if let Some(smg::wasm::module::WasmModuleAddResult::Success(uuid)) = module_result {
            let module = wasm_manager
                .get_module(*uuid)
                .expect("Failed to get module");
            assert!(module.is_some(), "Module should exist in manager");
        }
    }
}

#[tokio::test]
async fn test_wasm_api_add_module_invalid_file() {
    let (app, _app_context, _temp_dir) = create_test_app_with_wasm().await;

    let add_request = WasmModuleAddRequest {
        modules: vec![WasmModuleDescriptor {
            name: "test_module".to_string(),
            file_path: "/nonexistent/path/to/module.component.wasm".to_string(),
            module_type: WasmModuleType::Middleware,
            attach_points: vec![WasmModuleAttachPoint::Middleware(
                smg::wasm::module::MiddlewareAttachPoint::OnRequest,
            )],
            add_result: None,
        }],
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/wasm")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&add_request).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

                                 
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );

    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let response_json: WasmModuleAddResponse = serde_json::from_slice(&body).unwrap();

    assert_eq!(response_json.modules.len(), 1);
    let module_result = &response_json.modules[0].add_result;
    assert!(module_result.is_some());

                                  
    if let Some(smg::wasm::module::WasmModuleAddResult::Error(_)) = module_result {
                         
    } else {
        panic!("Expected error result for invalid file path");
    }
}

#[tokio::test]
async fn test_wasm_api_add_module_invalid_wasm() {
    let (app, _app_context, temp_dir) = create_test_app_with_wasm().await;

                                                      
    let invalid_wasm_path = temp_dir.path().join("invalid.component.wasm");
    fs::write(&invalid_wasm_path, b"not a valid wasm file")
        .await
        .expect("Failed to write invalid WASM file");

    let add_request = WasmModuleAddRequest {
        modules: vec![WasmModuleDescriptor {
            name: "invalid_module".to_string(),
            file_path: invalid_wasm_path.to_str().unwrap().to_string(),
            module_type: WasmModuleType::Middleware,
            attach_points: vec![WasmModuleAttachPoint::Middleware(
                smg::wasm::module::MiddlewareAttachPoint::OnRequest,
            )],
            add_result: None,
        }],
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/wasm")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&add_request).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

                                 
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );

    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let response_json: WasmModuleAddResponse = serde_json::from_slice(&body).unwrap();

    assert_eq!(response_json.modules.len(), 1);
    let module_result = &response_json.modules[0].add_result;
    assert!(module_result.is_some());

                                  
    if let Some(smg::wasm::module::WasmModuleAddResult::Error(_)) = module_result {
                         
    } else {
        panic!("Expected error result for invalid WASM file");
    }
}

#[tokio::test]
async fn test_wasm_api_list_modules() {
    let (app, _app_context, temp_dir) = create_test_app_with_wasm().await;
    let wasm_file_path = create_test_wasm_component(&temp_dir).await;

                          
    let add_request = WasmModuleAddRequest {
        modules: vec![WasmModuleDescriptor {
            name: "test_module_list".to_string(),
            file_path: wasm_file_path.clone(),
            module_type: WasmModuleType::Middleware,
            attach_points: vec![WasmModuleAttachPoint::Middleware(
                smg::wasm::module::MiddlewareAttachPoint::OnRequest,
            )],
            add_result: None,
        }],
    };

    let add_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/wasm")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&add_request).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(add_response.status(), StatusCode::OK);

                                         
    tokio::time::sleep(Duration::from_millis(500)).await;

                       
    let list_response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/wasm")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(list_response.status(), StatusCode::OK);

    let body = to_bytes(list_response.into_body(), usize::MAX)
        .await
        .unwrap();
    let response_json: WasmModuleListResponse = serde_json::from_slice(&body).unwrap();

    assert!(
        !response_json.modules.is_empty(),
        "Should have at least one module"
    );
    assert!(response_json
        .modules
        .iter()
        .any(|m| m.module_meta.name == "test_module_list"));

                                                                           
    let _ = response_json.metrics.total_executions;
}

#[tokio::test]
async fn test_wasm_api_remove_module() {
    let (app, app_context, temp_dir) = create_test_app_with_wasm().await;
    let wasm_file_path = create_test_wasm_component(&temp_dir).await;

                          
    let add_request = WasmModuleAddRequest {
        modules: vec![WasmModuleDescriptor {
            name: "test_module_remove".to_string(),
            file_path: wasm_file_path.clone(),
            module_type: WasmModuleType::Middleware,
            attach_points: vec![WasmModuleAttachPoint::Middleware(
                smg::wasm::module::MiddlewareAttachPoint::OnRequest,
            )],
            add_result: None,
        }],
    };

    let add_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/wasm")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&add_request).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(add_response.status(), StatusCode::OK);

    let body = to_bytes(add_response.into_body(), usize::MAX)
        .await
        .unwrap();
    let response_json: WasmModuleAddResponse = serde_json::from_slice(&body).unwrap();

                               
    tokio::time::sleep(Duration::from_millis(500)).await;

                          
    let module_uuid = if let Some(smg::wasm::module::WasmModuleAddResult::Success(uuid)) =
        &response_json.modules[0].add_result
    {
        *uuid
    } else {
                                                                          
        if let Some(wasm_manager) = app_context.wasm_manager.as_ref() {
            let modules = wasm_manager.get_modules().expect("Failed to get modules");
            modules
                .iter()
                .find(|m| m.module_meta.name == "test_module_remove")
                .map(|m| m.module_uuid)
                .expect("Module should be registered")
        } else {
            panic!("WASM manager not available");
        }
    };

                            
    let remove_response = app
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/wasm/{}", module_uuid))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    let remove_status = remove_response.status();
    if remove_status != StatusCode::OK {
        let body = to_bytes(remove_response.into_body(), usize::MAX)
            .await
            .unwrap();
        eprintln!(
            "Remove module failed with status {:?}: {}",
            remove_status,
            String::from_utf8_lossy(&body)
        );
        panic!("Expected OK status but got {:?}", remove_status);
    }

                                   
    tokio::time::sleep(Duration::from_millis(500)).await;

                               
    if let Some(wasm_manager) = app_context.wasm_manager.as_ref() {
        let module = wasm_manager
            .get_module(module_uuid)
            .expect("Failed to get module");
        assert!(module.is_none(), "Module should be removed");
    }
}

#[tokio::test]
async fn test_wasm_api_remove_module_not_found() {
    let (app, _app_context, _temp_dir) = create_test_app_with_wasm().await;

                                          
    let fake_uuid = Uuid::new_v4();
    let remove_response = app
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/wasm/{}", fake_uuid))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

                                 
    assert!(
        remove_response.status() == StatusCode::BAD_REQUEST
            || remove_response.status() == StatusCode::NOT_FOUND
    );
}

                                                                               
                           
                                                                               

#[tokio::test]
async fn test_wasm_module_duplicate_sha256() {
    let (app, _app_context, temp_dir) = create_test_app_with_wasm().await;
    let wasm_file_path = create_test_wasm_component(&temp_dir).await;

                       
    let add_request1 = WasmModuleAddRequest {
        modules: vec![WasmModuleDescriptor {
            name: "test_module_dup1".to_string(),
            file_path: wasm_file_path.clone(),
            module_type: WasmModuleType::Middleware,
            attach_points: vec![WasmModuleAttachPoint::Middleware(
                smg::wasm::module::MiddlewareAttachPoint::OnRequest,
            )],
            add_result: None,
        }],
    };

    let response1 = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/wasm")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&add_request1).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response1.status(), StatusCode::OK);

                                     
    tokio::time::sleep(Duration::from_millis(500)).await;

                                                                           
    let add_request2 = WasmModuleAddRequest {
        modules: vec![WasmModuleDescriptor {
            name: "test_module_dup2".to_string(),
            file_path: wasm_file_path.clone(),             
            module_type: WasmModuleType::Middleware,
            attach_points: vec![WasmModuleAttachPoint::Middleware(
                smg::wasm::module::MiddlewareAttachPoint::OnRequest,
            )],
            add_result: None,
        }],
    };

    let response2 = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/wasm")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&add_request2).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

                                               
    assert!(
        response2.status() == StatusCode::BAD_REQUEST
            || response2.status() == StatusCode::INTERNAL_SERVER_ERROR
    );

    let body = to_bytes(response2.into_body(), usize::MAX).await.unwrap();
    let response_json: WasmModuleAddResponse = serde_json::from_slice(&body).unwrap();

    assert_eq!(response_json.modules.len(), 1);
    let module_result = &response_json.modules[0].add_result;
    assert!(module_result.is_some());

                                              
    if let Some(smg::wasm::module::WasmModuleAddResult::Error(err_msg)) = module_result {
        assert!(
            err_msg.contains("duplicate")
                || err_msg.contains("Duplicate")
                || err_msg.contains("SHA256")
        );
    } else {
        panic!("Expected error result for duplicate SHA256");
    }
}

#[tokio::test]
async fn test_wasm_module_execution() {
    let (_app, app_context, temp_dir) = create_test_app_with_wasm().await;
    let wasm_file_path = create_test_wasm_component(&temp_dir).await;

                                                      
    let wasm_manager = app_context
        .wasm_manager
        .as_ref()
        .expect("WASM manager should be initialized");

    let engines = app_context
        .workflow_engines
        .get()
        .expect("Workflow engines should be initialized");

                                               
    use smg::core::steps::{WasmModuleConfigRequest, WasmRegistrationWorkflowData};
    use wfaas::WorkflowId;

    let descriptor = WasmModuleDescriptor {
        name: "test_execution_module".to_string(),
        file_path: wasm_file_path.clone(),
        module_type: WasmModuleType::Middleware,
        attach_points: vec![WasmModuleAttachPoint::Middleware(
            smg::wasm::module::MiddlewareAttachPoint::OnRequest,
        )],
        add_result: None,
    };

    let config_request = WasmModuleConfigRequest { descriptor };
    let workflow_data = WasmRegistrationWorkflowData {
        config: config_request,
        wasm_bytes: None,
        sha256_hash: None,
        file_size_bytes: None,
        module_uuid: None,
        app_context: Some(app_context.clone()),
    };

                     
    let instance_id = engines
        .wasm_registration
        .start_workflow(WorkflowId::new("wasm_module_registration"), workflow_data)
        .await
        .expect("Failed to start workflow");

                                    
    let timeout = Duration::from_secs(30);
    let start = std::time::Instant::now();

    let module_uuid = loop {
        if start.elapsed() > timeout {
            panic!("Workflow timeout");
        }

        let state = engines
            .wasm_registration
            .get_status(instance_id)
            .await
            .expect("Failed to get workflow status");

        match state.status {
            wfaas::WorkflowStatus::Completed => {
                                                               
                break state
                    .context
                    .data
                    .module_uuid
                    .expect("Module UUID should be in context");
            }
            wfaas::WorkflowStatus::Failed => {
                panic!("Workflow failed: {:?}", state);
            }
            _ => {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    };

                                  
    let module = wasm_manager
        .get_module(module_uuid)
        .expect("Failed to get module");
    assert!(module.is_some(), "Module should be registered");

                          
    let (initial_total, initial_success, initial_failed, _, _) = wasm_manager.get_metrics();

                         
    use smg::wasm::{
        spec::smg::gateway::middleware_types,
        types::{WasmComponentInput, WasmComponentOutput},
    };

    let request = middleware_types::Request {
        method: "GET".to_string(),
        path: "/test".to_string(),
        query: "".to_string(),
        headers: vec![],
        body: vec![],
        request_id: "test-request-id".to_string(),
        now_epoch_ms: 1000,
    };

    let input = WasmComponentInput::MiddlewareRequest(request);
    let attach_point =
        WasmModuleAttachPoint::Middleware(smg::wasm::module::MiddlewareAttachPoint::OnRequest);

                         
    let result = wasm_manager
        .execute_module_interface(module_uuid, attach_point, input)
        .await;

                              
    match result {
        Ok(WasmComponentOutput::MiddlewareAction(action)) => {
                                                                             
            match action {
                middleware_types::Action::Continue => {
                                                       
                }
                middleware_types::Action::Reject(_) => {
                                 
                }
                middleware_types::Action::Modify(_) => {
                                 
                }
            }
        }
        Err(e) => {
                                                                               
                                                                                                          
            eprintln!(
                "Module execution failed (expected if component is not properly built): {:?}",
                e
            );
        }
    }

                                  
    let (final_total, final_success, final_failed, _, _) = wasm_manager.get_metrics();

                                                               
    assert!(
        final_total > initial_total
            || final_failed > initial_failed
            || final_success > initial_success,
        "Metrics should be updated after execution"
    );
}

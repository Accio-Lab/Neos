use std::sync::{Arc, OnceLock};

use axum::Router;
use data_connector::{
    MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
};
use reqwest::Client;
use smg::{
    app_context::AppContext,
    config::RouterConfig,
    core::{
        BasicWorkerBuilder, LoadMonitor, ModelCard, RuntimeType, Worker, WorkerRegistry, WorkerType,
    },
    middleware::{AuthConfig, TokenBucket},
    policies::PolicyRegistry,
    routers::RouterTrait,
    server::{build_app, AppState},
    tokenizer::registry::TokenizerRegistry,
};
use smg_mcp::{McpConfig, McpManager};

/// Create a test Axum application using the actual server's build_app function
#[allow(dead_code)]
pub fn create_test_app(
    router: Arc<dyn RouterTrait>,
    client: Client,
    router_config: &RouterConfig,
) -> Router {
                              
    let rate_limiter = match router_config.max_concurrent_requests {
        n if n <= 0 => None,
        n => {
            let rate_limit_tokens = router_config
                .rate_limit_tokens_per_second
                .filter(|&t| t > 0)
                .unwrap_or(n);
            Some(Arc::new(TokenBucket::new(
                n as usize,
                rate_limit_tokens as usize,
            )))
        }
    };

                            
    let worker_registry = Arc::new(WorkerRegistry::new());
    let policy_registry = Arc::new(PolicyRegistry::new(router_config.policy.clone()));

                                  
    let response_storage = Arc::new(MemoryResponseStorage::new());
    let conversation_storage = Arc::new(MemoryConversationStorage::new());
    let conversation_item_storage = Arc::new(MemoryConversationItemStorage::new());

                              
    let load_monitor = Some(Arc::new(LoadMonitor::new(
        worker_registry.clone(),
        policy_registry.clone(),
        client.clone(),
        router_config.worker_startup_check_interval_secs,
    )));

                                                                      
    let worker_job_queue = Arc::new(OnceLock::new());
    let workflow_engines = Arc::new(OnceLock::new());

                                              
    let app_context = Arc::new(
        AppContext::builder()
            .router_config(router_config.clone())
            .client(client)
            .rate_limiter(rate_limiter)
            .tokenizer_registry(Arc::new(TokenizerRegistry::new()))             
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
            .build()
            .unwrap(),
    );

                                                       
    let app_state = Arc::new(AppState {
        router,
        context: app_context,
        concurrency_queue_tx: None,
        router_manager: None,
        mesh_handler: None,
        mesh_sync_manager: None,
    });

                                                                   
    let request_id_headers = router_config.request_id_headers.clone().unwrap_or_else(|| {
        vec![
            "x-request-id".to_string(),
            "x-correlation-id".to_string(),
            "x-trace-id".to_string(),
            "request-id".to_string(),
        ]
    });

                                            
    let auth_config = AuthConfig {
        api_key: router_config.api_key.clone(),
    };

                                                 
    build_app(
        app_state,
        auth_config,
        None,                                   
        router_config.max_payload_size,
        request_id_headers,
        router_config.cors_allowed_origins.clone(),
    )
}

/// Create a test Axum application with an existing AppContext
#[allow(dead_code)]
pub fn create_test_app_with_context(
    router: Arc<dyn RouterTrait>,
    app_context: Arc<AppContext>,
) -> Router {
                                                       
    let app_state = Arc::new(AppState {
        router,
        context: app_context.clone(),
        concurrency_queue_tx: None,
        router_manager: None,
        mesh_handler: None,
        mesh_sync_manager: None,
    });

                                  
    let router_config = &app_context.router_config;

                                                                   
    let request_id_headers = router_config.request_id_headers.clone().unwrap_or_else(|| {
        vec![
            "x-request-id".to_string(),
            "x-correlation-id".to_string(),
            "x-trace-id".to_string(),
            "request-id".to_string(),
        ]
    });

                                            
    let auth_config = AuthConfig {
        api_key: router_config.api_key.clone(),
    };

                                                 
    build_app(
        app_state,
        auth_config,
        None,                                   
        router_config.max_payload_size,
        request_id_headers,
        router_config.cors_allowed_origins.clone(),
    )
}

/// Create a minimal test AppContext for unit tests
#[allow(dead_code)]
pub async fn create_test_app_context() -> Arc<AppContext> {
    let router_config = RouterConfig::default();
    let client = Client::new();

                                 
    let worker_job_queue = Arc::new(OnceLock::new());
    let workflow_engines = Arc::new(OnceLock::new());

                                               
    let mcp_manager_lock = Arc::new(OnceLock::new());
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
    mcp_manager_lock.set(Arc::new(mcp_manager)).ok();

                            
    let worker_registry = Arc::new(WorkerRegistry::new());
    let policy_registry = Arc::new(PolicyRegistry::new(router_config.policy.clone()));

                                  
    let response_storage = Arc::new(MemoryResponseStorage::new());
    let conversation_storage = Arc::new(MemoryConversationStorage::new());
    let conversation_item_storage = Arc::new(MemoryConversationItemStorage::new());

    Arc::new(
        AppContext::builder()
            .router_config(router_config)
            .client(client)
            .rate_limiter(None)
            .tokenizer_registry(Arc::new(TokenizerRegistry::new()))
            .reasoning_parser_factory(None)
            .tool_parser_factory(None)
            .worker_registry(worker_registry)
            .policy_registry(policy_registry)
            .response_storage(response_storage)
            .conversation_storage(conversation_storage)
            .conversation_item_storage(conversation_item_storage)
            .load_monitor(None)
            .worker_job_queue(worker_job_queue)
            .workflow_engines(workflow_engines)
            .mcp_manager(mcp_manager_lock)
            .build()
            .unwrap(),
    )
}

/// Register an external worker (OpenAI-compatible API endpoint) in the test AppContext.
///
/// This is used by tests that need to test the OpenAI router, which expects
/// workers to be registered in the WorkerRegistry before routing requests.
///
/// # Arguments
/// * `ctx` - The AppContext to register the worker in
/// * `url` - The base URL of the external API endpoint
/// * `models` - Optional list of model IDs this worker supports. If empty, uses "gpt-3.5-turbo" as default.
#[allow(dead_code)]
pub fn register_external_worker(ctx: &Arc<AppContext>, url: &str, models: Option<Vec<&str>>) {
    let model_list: Vec<ModelCard> = models
        .unwrap_or_else(|| vec!["gpt-3.5-turbo"])
        .into_iter()
        .map(ModelCard::new)
        .collect();

    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .runtime_type(RuntimeType::External)
            .models(model_list)
            .build(),
    );

    ctx.worker_registry.register(worker);
}

/// Register an external worker with a custom model card that has aliases.
///
/// # Arguments
/// * `ctx` - The AppContext to register the worker in
/// * `url` - The base URL of the external API endpoint
/// * `model_card` - A fully configured ModelCard with aliases, provider, etc.
#[allow(dead_code)]
pub fn register_external_worker_with_card(ctx: &Arc<AppContext>, url: &str, model_card: ModelCard) {
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .runtime_type(RuntimeType::External)
            .model(model_card)
            .build(),
    );

    ctx.worker_registry.register(worker);
}

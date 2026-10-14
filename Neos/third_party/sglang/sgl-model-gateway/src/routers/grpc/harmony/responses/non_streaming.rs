//! Non-streaming Harmony Responses API implementation

use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::response::Response;
use serde_json::{json, to_string};
use tracing::{debug, error, warn};

use super::{
    common::{
        build_mcp_tool_names_set, build_next_request_with_tools, inject_mcp_metadata,
        load_previous_messages, McpCallTracking,
    },
    execution::{convert_mcp_tools_to_response_tools, execute_mcp_tools, ToolResult},
};
use crate::{
    observability::metrics::Metrics,
    protocols::{
        common::{ToolCall, Usage},
        responses::{
            OutputTokensDetails, ResponseContentPart, ResponseOutputItem, ResponseReasoningContent,
            ResponseStatus, ResponseUsage, ResponsesRequest, ResponsesResponse, ResponsesUsage,
        },
    },
    routers::{
        error,
        grpc::{
            common::responses::{
                ensure_mcp_connection, persist_response_if_needed, ResponsesContext,
            },
            harmony::processor::ResponsesIterationResult,
        },
        mcp_utils::{extract_server_label, DEFAULT_MAX_ITERATIONS},
    },
};

/// Execute Harmony Responses API request with multi-turn MCP tool support
///
/// This function orchestrates the multi-turn conversation flow:
/// 1. Execute request through full pipeline
/// 2. Check for tool calls in commentary channel
/// 3. If tool calls found:
///    - Execute MCP tools
///    - Build next request with tool results
///    - Repeat from step 1 (full pipeline re-execution)
/// 4. If no tool calls, return final response
pub(crate) async fn serve_harmony_responses(
    ctx: &ResponsesContext,
    request: ResponsesRequest,
) -> Result<ResponsesResponse, Response> {
                                    
    let original_request = request.clone();

                                                                        
    let current_request = load_previous_messages(ctx, request).await?;

                                                                 
    let (has_mcp_tools, server_keys) =
        ensure_mcp_connection(&ctx.mcp_manager, current_request.tools.as_deref()).await?;

                                         
    {
        let mut servers = ctx.requested_servers.write().unwrap();
        *servers = server_keys;
    }

    let response = if has_mcp_tools {
        execute_with_mcp_loop(ctx, current_request).await?
    } else {
                                                                                     
        execute_without_mcp_loop(ctx, current_request).await?
    };

                                                
    persist_response_if_needed(
        ctx.conversation_storage.clone(),
        ctx.conversation_item_storage.clone(),
        ctx.response_storage.clone(),
        &response,
        &original_request,
    )
    .await;

    Ok(response)
}

/// Execute Harmony Responses with MCP tool loop
///
/// Automatically executes MCP tools in a loop until no more tool calls or max iterations
async fn execute_with_mcp_loop(
    ctx: &ResponsesContext,
    mut current_request: ResponsesRequest,
) -> Result<ResponsesResponse, Response> {
    let mut iteration_count = 0;

                                              
    let server_label = extract_server_label(current_request.tools.as_deref(), "sglang-mcp");
    let mut mcp_tracking = McpCallTracking::new(server_label.clone());

                                                   
    let max_tool_calls = current_request.max_tool_calls.map(|n| n as usize);

                                                                         
    let mcp_tools = {
        let servers = ctx.requested_servers.read().unwrap();
        ctx.mcp_manager.list_tools_for_servers(&servers)
    };
    if !mcp_tools.is_empty() {
        let mcp_response_tools = convert_mcp_tools_to_response_tools(&mcp_tools);

        let mut all_tools = current_request.tools.clone().unwrap_or_default();
        all_tools.extend(mcp_response_tools);
        current_request.tools = Some(all_tools);

        debug!(
            mcp_tool_count = mcp_tools.len(),
            total_tool_count = current_request.tools.as_ref().map(|t| t.len()).unwrap_or(0),
            "MCP client available - added static MCP tools to Harmony Responses request"
        );
    }

    loop {
        iteration_count += 1;

                                            
        Metrics::record_mcp_tool_iteration(&current_request.model);

                                               
        if iteration_count > DEFAULT_MAX_ITERATIONS {
            error!(
                function = "execute_with_mcp_loop",
                iteration_count = iteration_count,
                max_iterations = DEFAULT_MAX_ITERATIONS,
                "Maximum tool iterations exceeded"
            );
            return Err(error::internal_error(
                "tool_iterations_exceeded",
                format!(
                    "Maximum tool iterations ({}) exceeded",
                    DEFAULT_MAX_ITERATIONS
                ),
            ));
        }

        debug!(
            iteration = iteration_count,
            "Harmony Responses serving iteration"
        );

                                        
        let iteration_result = ctx
            .pipeline
            .execute_harmony_responses(&current_request, ctx)
            .await?;

        match iteration_result {
            ResponsesIterationResult::ToolCallsFound {
                tool_calls,
                analysis,
                partial_text,
                usage,
                request_id,
            } => {
                debug!(
                    tool_call_count = tool_calls.len(),
                    has_analysis = analysis.is_some(),
                    partial_text_len = partial_text.len(),
                    "Tool calls found - separating MCP and function tools"
                );

                                                                          
                let request_tools = current_request.tools.as_deref().unwrap_or(&[]);
                let mcp_tool_names = build_mcp_tool_names_set(request_tools);
                let (mcp_tool_calls, function_tool_calls): (Vec<_>, Vec<_>) = tool_calls
                    .into_iter()
                    .partition(|tc| mcp_tool_names.contains(tc.function.name.as_str()));

                debug!(
                    mcp_calls = mcp_tool_calls.len(),
                    function_calls = function_tool_calls.len(),
                    "Tool calls separated by type"
                );

                                                                               
                let effective_limit = match max_tool_calls {
                    Some(user_max) => user_max.min(DEFAULT_MAX_ITERATIONS),
                    None => DEFAULT_MAX_ITERATIONS,
                };

                                                                                   
                let total_calls_after = mcp_tracking.total_calls() + mcp_tool_calls.len();
                if total_calls_after > effective_limit {
                    warn!(
                        current_calls = mcp_tracking.total_calls(),
                        new_calls = mcp_tool_calls.len() + function_tool_calls.len(),
                        total_after = total_calls_after,
                        effective_limit = effective_limit,
                        user_max = ?max_tool_calls,
                        "Reached tool call limit - returning incomplete response"
                    );

                                                
                    let all_tool_calls: Vec<_> = mcp_tool_calls
                        .into_iter()
                        .chain(function_tool_calls)
                        .collect();

                                                                                             
                    let mut response = build_tool_response(
                        vec![],                                 
                        vec![],                          
                        all_tool_calls,                                                       
                        analysis,
                        partial_text,
                        usage,
                        request_id,
                        Arc::new(current_request),
                    );

                                                                
                    response.status = ResponseStatus::Completed;
                    response.incomplete_details = Some(json!({ "reason": "max_tool_calls" }));

                                                                     
                    if mcp_tracking.total_calls() > 0 {
                        inject_mcp_metadata(&mut response, &mcp_tracking, &mcp_tools);
                    }

                    return Ok(response);
                }

                                             
                let mcp_results = if !mcp_tool_calls.is_empty() {
                    execute_mcp_tools(
                        &ctx.mcp_manager,
                        &mcp_tool_calls,
                        &mut mcp_tracking,
                        &current_request.model,
                    )
                    .await?
                } else {
                    Vec::new()
                };

                                                                                 
                if !function_tool_calls.is_empty() {
                    debug!(
                        "Function tool calls present - exiting MCP loop and returning to caller"
                    );

                                                    
                                                               
                                                                                    
                                                                                              
                    let mut response = build_tool_response(
                        mcp_tool_calls,
                        mcp_results,
                        function_tool_calls,
                        analysis,
                        partial_text,
                        usage,
                        request_id,
                        Arc::new(current_request),
                    );

                                                                 
                    if mcp_tracking.total_calls() > 0 {
                        inject_mcp_metadata(&mut response, &mcp_tracking, &mcp_tools);
                    }

                    return Ok(response);
                }

                                                                    
                debug!("Only MCP tools - continuing loop with results");

                                                           
                current_request = build_next_request_with_tools(
                    current_request,
                    mcp_tool_calls,
                    mcp_results,
                    analysis,
                    partial_text,
                )
                .map_err(|e| *e)?;

                                                                                 
            }
            ResponsesIterationResult::Completed {
                mut response,
                usage,
            } => {
                debug!(
                    output_items = response.output.len(),
                    input_tokens = usage.prompt_tokens,
                    output_tokens = usage.completion_tokens,
                    "MCP loop completed - no more tool calls"
                );

                                                          
                inject_mcp_metadata(&mut response, &mcp_tracking, &mcp_tools);

                debug!(
                    mcp_calls = mcp_tracking.total_calls(),
                    output_items_after = response.output.len(),
                    "Injected MCP metadata into final response"
                );

                                                             
                return Ok(*response);
            }
        }
    }
}

/// Execute Harmony Responses without MCP loop (single execution)
///
/// For function tools or no tools - executes pipeline once and returns
async fn execute_without_mcp_loop(
    ctx: &ResponsesContext,
    current_request: ResponsesRequest,
) -> Result<ResponsesResponse, Response> {
    debug!("Executing Harmony Responses without MCP loop");

                            
    let iteration_result = ctx
        .pipeline
        .execute_harmony_responses(&current_request, ctx)
        .await?;

    match iteration_result {
        ResponsesIterationResult::ToolCallsFound {
            tool_calls,
            analysis,
            partial_text,
            usage,
            request_id,
        } => {
                                                                         
            debug!(
                tool_call_count = tool_calls.len(),
                "Function tool calls found - returning to caller"
            );

            Ok(build_tool_response(
                vec![],
                vec![],
                tool_calls,
                analysis,
                partial_text,
                usage,
                request_id,
                Arc::new(current_request),
            ))
        }
        ResponsesIterationResult::Completed { response, usage: _ } => {
                                                        
            debug!("No tool calls - returning completed response");
            Ok(*response)
        }
    }
}

/// Build ResponsesResponse with tool calls (MCP and/or function tools)
#[allow(clippy::too_many_arguments)]
fn build_tool_response(
    mcp_tool_calls: Vec<ToolCall>,
    mcp_results: Vec<ToolResult>,
    function_tool_calls: Vec<ToolCall>,
    analysis: Option<String>,                                        
    partial_text: String,                                       
    usage: Usage,
    request_id: String,
    responses_request: Arc<ResponsesRequest>,
) -> ResponsesResponse {
    let mut output: Vec<ResponseOutputItem> = Vec::new();

                                                   
    if let Some(analysis_text) = analysis {
        output.push(ResponseOutputItem::Reasoning {
            id: format!("reasoning_{}", request_id),
            summary: vec![],
            content: vec![ResponseReasoningContent::ReasoningText {
                text: analysis_text,
            }],
            status: Some("completed".to_string()),
        });
    }

                                                     
    if !partial_text.is_empty() {
        output.push(ResponseOutputItem::Message {
            id: format!("msg_{}", request_id),
            role: "assistant".to_string(),
            content: vec![ResponseContentPart::OutputText {
                text: partial_text,
                annotations: vec![],
                logprobs: None,
            }],
            status: "completed".to_string(),
        });
    }

                                                           
    for (tool_call, result) in mcp_tool_calls.iter().zip(mcp_results.iter()) {
        let output_str = to_string(&result.output).unwrap_or_else(|e| {
            format!("{{\"error\": \"Failed to serialize tool output: {}\"}}", e)
        });

        output.push(ResponseOutputItem::FunctionToolCall {
            id: tool_call.id.clone(),
            call_id: tool_call.id.clone(),
            name: tool_call.function.name.clone(),
            arguments: tool_call.function.arguments.clone().unwrap_or_default(),
            output: Some(output_str),
            status: if result.is_error {
                "failed"
            } else {
                "completed"
            }
            .to_string(),
        });
    }

                                                                     
    for tool_call in function_tool_calls {
        output.push(ResponseOutputItem::FunctionToolCall {
            id: tool_call.id.clone(),
            call_id: tool_call.id.clone(),
            name: tool_call.function.name.clone(),
            arguments: tool_call.function.arguments.clone().unwrap_or_default(),
            output: None,                               
            status: "completed".to_string(),
        });
    }

                                                    
    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    ResponsesResponse::builder(&request_id, &responses_request.model)
        .copy_from_request(&responses_request)
        .created_at(created_at)
        .status(ResponseStatus::Completed)
        .output(output)
        .usage(ResponsesUsage::Modern(ResponseUsage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            input_tokens_details: None,
            output_tokens_details: usage.completion_tokens_details.as_ref().and_then(|d| {
                d.reasoning_tokens.map(|tokens| OutputTokensDetails {
                    reasoning_tokens: tokens,
                })
            }),
        }))
        .build()
}

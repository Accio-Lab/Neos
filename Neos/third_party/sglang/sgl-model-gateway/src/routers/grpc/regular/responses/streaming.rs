//! Streaming execution for Regular Responses API
//!
//! This module handles streaming request execution:
//! - `execute_tool_loop_streaming` - MCP tool loop with streaming
//! - `convert_chat_stream_to_responses_stream` - Non-MCP streaming conversion
//! - Streaming accumulators for response building

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use axum::{
    body::Body,
    http::{self, header, StatusCode},
    response::Response,
};
use bytes::Bytes;
use data_connector::{ConversationItemStorage, ConversationStorage, ResponseStorage};
use futures_util::StreamExt;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, trace, warn};
use uuid::Uuid;

use super::{
    common::{
        build_next_request, convert_mcp_tools_to_chat_tools, extract_all_tool_calls_from_chat,
        prepare_chat_tools_and_choice, ExtractedToolCall, ToolLoopState,
    },
    conversions,
};
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    protocols::{
        chat::{
            ChatChoice, ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse,
            ChatCompletionStreamResponse,
        },
        common::{FunctionCallResponse, ToolCall, Usage, UsageInfo},
        responses::{
            ResponseContentPart, ResponseOutputItem, ResponseReasoningContent, ResponseStatus,
            ResponsesRequest, ResponsesResponse, ResponsesUsage,
        },
    },
    routers::{
        grpc::common::responses::{
            build_sse_response, persist_response_if_needed,
            streaming::{OutputItemType, ResponseStreamEventEmitter},
            ResponsesContext,
        },
        mcp_utils::{extract_server_label, DEFAULT_MAX_ITERATIONS},
    },
};

                                                                               
                         
                                                                               

/// Convert chat streaming response to responses streaming format
///
/// This function:
/// 1. Gets chat SSE stream from pipeline
/// 2. Intercepts and parses each SSE event
/// 3. Converts ChatCompletionStreamResponse → ResponsesResponse delta
/// 4. Accumulates response state for final persistence
/// 5. Emits transformed SSE events in responses format
pub(super) async fn convert_chat_stream_to_responses_stream(
    ctx: &ResponsesContext,
    chat_request: Arc<ChatCompletionRequest>,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    original_request: &ResponsesRequest,
) -> Response {
    debug!("Converting chat SSE stream to responses SSE format");

                                  
    let chat_response = ctx
        .pipeline
        .execute_chat(
            chat_request.clone(),
            headers,
            model_id,
            ctx.components.clone(),
        )
        .await;

                                      
    let (_parts, body) = chat_response.into_parts();

                                                
    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, std::io::Error>>();

                                                
    let original_request_clone = original_request.clone();
    let response_storage = ctx.response_storage.clone();
    let conversation_storage = ctx.conversation_storage.clone();
    let conversation_item_storage = ctx.conversation_item_storage.clone();

    tokio::spawn(async move {
        if let Err(e) = process_and_transform_sse_stream(
            body,
            original_request_clone,
            response_storage,
            conversation_storage,
            conversation_item_storage,
            tx.clone(),
        )
        .await
        {
            warn!("Error transforming SSE stream: {}", e);
            let error_event = json!({
                "error": {
                    "message": e,
                    "type": "stream_error"
                }
            });
            let _ = tx.send(Ok(Bytes::from(format!("data: {}\n\n", error_event))));
        }

                                  
        let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
    });

                                                 
    build_sse_response(rx)
}

/// Process chat SSE stream and transform to responses format
async fn process_and_transform_sse_stream(
    body: Body,
    original_request: ResponsesRequest,
    response_storage: Arc<dyn ResponseStorage>,
    conversation_storage: Arc<dyn ConversationStorage>,
    conversation_item_storage: Arc<dyn ConversationItemStorage>,
    tx: mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
) -> Result<(), String> {
                                            
    let mut accumulator = StreamingResponseAccumulator::new(&original_request);

                                                           
    let response_id = format!("resp_{}", Uuid::new_v4());
    let model = original_request.model.clone();
    let created_at = chrono::Utc::now().timestamp() as u64;
    let mut event_emitter = ResponseStreamEventEmitter::new(response_id, model, created_at);
    event_emitter.set_original_request(original_request.clone());

                                                                    
    let event = event_emitter.emit_created();
    event_emitter
        .send_event(&event, &tx)
        .map_err(|_| "Failed to send response.created event".to_string())?;

    let event = event_emitter.emit_in_progress();
    event_emitter
        .send_event(&event, &tx)
        .map_err(|_| "Failed to send response.in_progress event".to_string())?;

                                  
    let mut stream = body.into_data_stream();

                                                                 
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| format!("Stream read error: {}", e))?;

                                  
        let event_str = String::from_utf8_lossy(&chunk);
        let event = event_str.trim();

                                  
        if event == "data: [DONE]" {
            break;
        }

                                                                       
        if let Some(json_str) = event.strip_prefix("data: ") {
            let json_str = json_str.trim();

                                                           
            match serde_json::from_str::<ChatCompletionStreamResponse>(json_str) {
                Ok(chat_chunk) => {
                                         
                    accumulator.process_chunk(&chat_chunk);

                                                                                       
                    event_emitter.process_chunk(&chat_chunk, &tx)?;
                }
                Err(_) => {
                                                                                  
                    debug!("Non-chunk SSE event, passing through: {}", event);
                    if tx.send(Ok(Bytes::from(format!("{}\n\n", event)))).is_err() {
                        return Err("Client disconnected".to_string());
                    }
                }
            }
        }
    }

                                                                 
    let usage_json = accumulator.usage.as_ref().map(|u| {
        let mut usage_obj = json!({
            "input_tokens": u.prompt_tokens,
            "output_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens
        });

                                              
        if let Some(details) = &u.completion_tokens_details {
            if let Some(reasoning_tokens) = details.reasoning_tokens {
                usage_obj["output_tokens_details"] =
                    json!({ "reasoning_tokens": reasoning_tokens });
            }
        }

        usage_obj
    });

    let completed_event = event_emitter.emit_completed(usage_json.as_ref());
    event_emitter.send_event(&completed_event, &tx)?;

                                                
    let final_response = accumulator.finalize();
    persist_response_if_needed(
        conversation_storage,
        conversation_item_storage,
        response_storage,
        &final_response,
        &original_request,
    )
    .await;

    Ok(())
}

/// Response accumulator for streaming responses (non-MCP path)
struct StreamingResponseAccumulator {
                        
    response_id: String,
    model: String,
    created_at: i64,

                          
    content_buffer: String,
    reasoning_buffer: String,
    tool_calls: Vec<ResponseOutputItem>,

                       
    finish_reason: Option<String>,
    usage: Option<Usage>,

                                                       
    original_request: ResponsesRequest,
}

impl StreamingResponseAccumulator {
    fn new(original_request: &ResponsesRequest) -> Self {
        Self {
            response_id: String::new(),
            model: String::new(),
            created_at: 0,
            content_buffer: String::new(),
            reasoning_buffer: String::new(),
            tool_calls: Vec::new(),
            finish_reason: None,
            usage: None,
            original_request: original_request.clone(),
        }
    }

    fn process_chunk(&mut self, chunk: &ChatCompletionStreamResponse) {
                                             
        if self.response_id.is_empty() {
            self.response_id = chunk.id.clone();
            self.model = chunk.model.clone();
            self.created_at = chunk.created as i64;
        }

                                                                   
        if let Some(choice) = chunk.choices.first() {
                                 
            if let Some(content) = &choice.delta.content {
                self.content_buffer.push_str(content);
            }

                                   
            if let Some(reasoning) = &choice.delta.reasoning_content {
                self.reasoning_buffer.push_str(reasoning);
            }

                                       
            if let Some(tool_call_deltas) = &choice.delta.tool_calls {
                for delta in tool_call_deltas {
                                                                       
                    let index = delta.index as usize;

                                                       
                    while self.tool_calls.len() <= index {
                        self.tool_calls.push(ResponseOutputItem::FunctionToolCall {
                            id: String::new(),
                            call_id: String::new(),
                            name: String::new(),
                            arguments: String::new(),
                            output: None,
                            status: "in_progress".to_string(),
                        });
                    }

                                                         
                    if let ResponseOutputItem::FunctionToolCall {
                        id,
                        name,
                        arguments,
                        ..
                    } = &mut self.tool_calls[index]
                    {
                        if let Some(delta_id) = &delta.id {
                            id.push_str(delta_id);
                        }
                        if let Some(function) = &delta.function {
                            if let Some(delta_name) = &function.name {
                                name.push_str(delta_name);
                            }
                            if let Some(delta_args) = &function.arguments {
                                arguments.push_str(delta_args);
                            }
                        }
                    }
                }
            }

                                   
            if let Some(reason) = &choice.finish_reason {
                self.finish_reason = Some(reason.clone());
            }
        }

                       
        if let Some(usage) = &chunk.usage {
            self.usage = Some(usage.clone());
        }
    }

    fn finalize(self) -> ResponsesResponse {
        let mut output: Vec<ResponseOutputItem> = Vec::new();

                                         
        if !self.content_buffer.is_empty() {
            output.push(ResponseOutputItem::Message {
                id: format!("msg_{}", self.response_id),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text: self.content_buffer,
                    annotations: vec![],
                    logprobs: None,
                }],
                status: "completed".to_string(),
            });
        }

                                   
        if !self.reasoning_buffer.is_empty() {
            output.push(ResponseOutputItem::Reasoning {
                id: format!("reasoning_{}", self.response_id),
                summary: vec![],
                content: vec![ResponseReasoningContent::ReasoningText {
                    text: self.reasoning_buffer,
                }],
                status: Some("completed".to_string()),
            });
        }

                         
        output.extend(self.tool_calls);

                                 
        let status = match self.finish_reason.as_deref() {
            Some("stop") | Some("length") => ResponseStatus::Completed,
            Some("tool_calls") => ResponseStatus::InProgress,
            Some("failed") | Some("error") => ResponseStatus::Failed,
            _ => ResponseStatus::Completed,
        };

                        
        let usage = self.usage.as_ref().map(|u| {
            let usage_info = UsageInfo {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
                reasoning_tokens: u
                    .completion_tokens_details
                    .as_ref()
                    .and_then(|d| d.reasoning_tokens),
                prompt_tokens_details: None,
            };
            ResponsesUsage::Classic(usage_info)
        });

        ResponsesResponse::builder(&self.response_id, &self.model)
            .copy_from_request(&self.original_request)
            .created_at(self.created_at)
            .status(status)
            .output(output)
            .maybe_usage(usage)
            .build()
    }
}

                                                                               
                     
                                                                               

/// Execute MCP tool loop with streaming support
///
/// This streams each iteration's response to the client while accumulating
/// to check for tool calls. If tool calls are found, executes them and
/// continues with the next streaming iteration.
pub(super) async fn execute_tool_loop_streaming(
    ctx: &ResponsesContext,
    current_request: ResponsesRequest,
    original_request: &ResponsesRequest,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
) -> Response {
                                    
    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, std::io::Error>>();

                                     
    let ctx_clone = ctx.clone();
    let original_request_clone = original_request.clone();

                                          
    tokio::spawn(async move {
        let result = execute_tool_loop_streaming_internal(
            &ctx_clone,
            current_request,
            &original_request_clone,
            headers,
            model_id,
            tx.clone(),
        )
        .await;

        if let Err(e) = result {
            warn!("Streaming tool loop error: {}", e);
            let error_event = json!({
                "error": {
                    "message": e,
                    "type": "tool_loop_error"
                }
            });
            let _ = tx.send(Ok(Bytes::from(format!("data: {}\n\n", error_event))));
        }

                      
        let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
    });

                         
    let stream = UnboundedReceiverStream::new(rx);
    let body = Body::from_stream(stream);

    let mut response = Response::builder()
        .status(StatusCode::OK)
        .body(body)
        .unwrap();

    response.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("text/event-stream"),
    );
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        header::HeaderValue::from_static("no-cache"),
    );
    response.headers_mut().insert(
        header::CONNECTION,
        header::HeaderValue::from_static("keep-alive"),
    );

    response
}

/// Internal streaming tool loop implementation
async fn execute_tool_loop_streaming_internal(
    ctx: &ResponsesContext,
    mut current_request: ResponsesRequest,
    original_request: &ResponsesRequest,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    tx: mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
) -> Result<(), String> {
                                                       
    let server_label = extract_server_label(original_request.tools.as_deref(), "request-mcp");

    let mut state = ToolLoopState::new(original_request.input.clone(), server_label.clone());
    let max_tool_calls = original_request.max_tool_calls.map(|n| n as usize);

                                    
    let response_id = format!("resp_{}", Uuid::new_v4());
    let model = current_request.model.clone();
    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let mut emitter = ResponseStreamEventEmitter::new(response_id, model.clone(), created_at);
    emitter.set_original_request(original_request.clone());

                                                                    
    let event = emitter.emit_created();
    emitter.send_event(&event, &tx)?;
    let event = emitter.emit_in_progress();
    emitter.send_event(&event, &tx)?;

                                                                          
    let mcp_tools = ctx.mcp_manager.list_tools();
    let mcp_chat_tools = convert_mcp_tools_to_chat_tools(&mcp_tools);
    trace!(
        "Streaming: Converted {} MCP tools to chat format",
        mcp_chat_tools.len()
    );

                                                       
    let mut mcp_list_tools_emitted = false;

    loop {
        state.iteration += 1;

                                            
        Metrics::record_mcp_tool_iteration(&model);

        if state.iteration > DEFAULT_MAX_ITERATIONS {
            return Err(format!(
                "Tool loop exceeded maximum iterations ({})",
                DEFAULT_MAX_ITERATIONS
            ));
        }

        trace!("Streaming MCP tool loop iteration {}", state.iteration);

                                                                                   
        if !mcp_list_tools_emitted {
            let (output_index, item_id) =
                emitter.allocate_output_index(OutputItemType::McpListTools);

                                                  
            let tool_items: Vec<_> = mcp_tools
                .iter()
                .map(|t| {
                    json!({
                        "name": t.name,
                        "description": t.description,
                        "input_schema": Value::Object((*t.input_schema).clone())
                    })
                })
                .collect();

                                        
            let item = json!({
                "id": item_id,
                "type": "mcp_list_tools",
                "server_label": state.server_label,
                "status": "in_progress",
                "tools": []
            });

                                     
            let event = emitter.emit_output_item_added(output_index, &item);
            emitter.send_event(&event, &tx)?;

                                              
            let event = emitter.emit_mcp_list_tools_in_progress(output_index);
            emitter.send_event(&event, &tx)?;

                                            
            let event = emitter.emit_mcp_list_tools_completed(output_index, &mcp_tools);
            emitter.send_event(&event, &tx)?;

                                             
            let item_done = json!({
                "id": item_id,
                "type": "mcp_list_tools",
                "server_label": state.server_label,
                "status": "completed",
                "tools": tool_items
            });

                                    
            let event = emitter.emit_output_item_done(output_index, &item_done);
            emitter.send_event(&event, &tx)?;

            emitter.complete_output_item(output_index);
            mcp_list_tools_emitted = true;
        }

                                  
        let mut chat_request = conversions::responses_to_chat(&current_request)
            .map_err(|e| format!("Failed to convert request: {}", e))?;

                                                                                         
        prepare_chat_tools_and_choice(&mut chat_request, &mcp_chat_tools, state.iteration);

                                 
        let response = ctx
            .pipeline
            .execute_chat(
                Arc::new(chat_request),
                headers.clone(),
                model_id.clone(),
                ctx.components.clone(),
            )
            .await;

                                                                                                 
                                                                                                          
        let accumulated_response =
            convert_and_accumulate_stream(response.into_body(), &mut emitter, &tx).await?;

                                                                            
        let tool_calls = extract_all_tool_calls_from_chat(&accumulated_response);

        if !tool_calls.is_empty() {
            trace!(
                "Tool loop iteration {}: found {} tool call(s)",
                state.iteration,
                tool_calls.len()
            );

                                                   
            let mcp_tool_names: std::collections::HashSet<&str> =
                mcp_tools.iter().map(|t| t.name.as_ref()).collect();
            let (mcp_tool_calls, function_tool_calls): (Vec<ExtractedToolCall>, Vec<_>) =
                tool_calls
                    .into_iter()
                    .partition(|tc| mcp_tool_names.contains(tc.name.as_str()));

            trace!(
                "Separated tool calls: {} MCP, {} function",
                mcp_tool_calls.len(),
                function_tool_calls.len()
            );

                                                                                                
            let effective_limit = match max_tool_calls {
                Some(user_max) => user_max.min(DEFAULT_MAX_ITERATIONS),
                None => DEFAULT_MAX_ITERATIONS,
            };

            if state.total_calls + mcp_tool_calls.len() > effective_limit {
                warn!(
                    "Reached tool call limit: {} + {} > {} (max_tool_calls={:?}, safety_limit={})",
                    state.total_calls,
                    mcp_tool_calls.len(),
                    effective_limit,
                    max_tool_calls,
                    DEFAULT_MAX_ITERATIONS
                );
                break;
            }

                                         
            for tool_call in mcp_tool_calls {
                state.total_calls += 1;

                trace!(
                    "Executing tool call {}/{}: {} (call_id: {})",
                    state.total_calls,
                    state.total_calls,
                    tool_call.name,
                    tool_call.call_id
                );

                                                               
                let (output_index, item_id) =
                    emitter.allocate_output_index(OutputItemType::McpCall);

                                              
                let item = json!({
                    "id": item_id,
                    "type": "mcp_call",
                    "name": tool_call.name,
                    "server_label": state.server_label,
                    "status": "in_progress",
                    "arguments": ""
                });

                                         
                let event = emitter.emit_output_item_added(output_index, &item);
                emitter.send_event(&event, &tx)?;

                                            
                let event = emitter.emit_mcp_call_in_progress(output_index, &item_id);
                emitter.send_event(&event, &tx)?;

                                                                                               
                let event = emitter.emit_mcp_call_arguments_delta(
                    output_index,
                    &item_id,
                    &tool_call.arguments,
                );
                emitter.send_event(&event, &tx)?;

                                               
                let event = emitter.emit_mcp_call_arguments_done(
                    output_index,
                    &item_id,
                    &tool_call.arguments,
                );
                emitter.send_event(&event, &tx)?;

                                                                                   
                trace!(
                    "Calling MCP tool '{}' with args: {}",
                    tool_call.name,
                    tool_call.arguments
                );
                let tool_start = Instant::now();
                let (output_str, success, error) = match ctx
                    .mcp_manager
                    .call_tool(tool_call.name.as_str(), tool_call.arguments.as_str())
                    .await
                {
                    Ok(result) => match serde_json::to_string(&result) {
                        Ok(output) => {
                                                      
                            let event = emitter.emit_mcp_call_completed(output_index, &item_id);
                            emitter.send_event(&event, &tx)?;

                                                              
                            let item_done = json!({
                                "id": item_id,
                                "type": "mcp_call",
                                "name": tool_call.name,
                                "server_label": state.server_label,
                                "status": "completed",
                                "arguments": tool_call.arguments,
                                "output": output
                            });

                                                    
                            let event = emitter.emit_output_item_done(output_index, &item_done);
                            emitter.send_event(&event, &tx)?;

                            emitter.complete_output_item(output_index);
                            (output, true, None)
                        }
                        Err(e) => {
                            let err = format!("Failed to serialize tool result: {}", e);
                            warn!("{}", err);
                                                   
                            let event = emitter.emit_mcp_call_failed(output_index, &item_id, &err);
                            emitter.send_event(&event, &tx)?;

                                                
                            let item_done = json!({
                                "id": item_id,
                                "type": "mcp_call",
                                "name": tool_call.name,
                                "server_label": state.server_label,
                                "status": "failed",
                                "arguments": tool_call.arguments,
                                "error": &err
                            });

                                                    
                            let event = emitter.emit_output_item_done(output_index, &item_done);
                            emitter.send_event(&event, &tx)?;

                            emitter.complete_output_item(output_index);
                            let error_json = json!({ "error": &err }).to_string();
                            (error_json, false, Some(err))
                        }
                    },
                    Err(err) => {
                        let err_str = format!("tool call failed: {}", err);
                        warn!("Tool execution failed: {}", err_str);
                                               
                        let event = emitter.emit_mcp_call_failed(output_index, &item_id, &err_str);
                        emitter.send_event(&event, &tx)?;

                                            
                        let item_done = json!({
                            "id": item_id,
                            "type": "mcp_call",
                            "name": tool_call.name,
                            "server_label": state.server_label,
                            "status": "failed",
                            "arguments": tool_call.arguments,
                            "error": &err_str
                        });

                                                
                        let event = emitter.emit_output_item_done(output_index, &item_done);
                        emitter.send_event(&event, &tx)?;

                        emitter.complete_output_item(output_index);
                        let error_json = json!({ "error": &err_str }).to_string();
                        (error_json, false, Some(err_str))
                    }
                };
                let tool_duration = tool_start.elapsed();

                                          
                Metrics::record_mcp_tool_duration(&model, &tool_call.name, tool_duration);
                Metrics::record_mcp_tool_call(
                    &model,
                    &tool_call.name,
                    if success {
                        metrics_labels::RESULT_SUCCESS
                    } else {
                        metrics_labels::RESULT_ERROR
                    },
                );

                                           
                state.record_call(
                    tool_call.call_id,
                    tool_call.name,
                    tool_call.arguments,
                    output_str,
                    success,
                    error,
                );
            }

                                                                              
            if !function_tool_calls.is_empty() {
                trace!(
                    "Found {} function tool call(s) - emitting events and exiting MCP loop",
                    function_tool_calls.len()
                );

                                                                        
                for tool_call in function_tool_calls {
                                                                             
                    let (output_index, item_id) =
                        emitter.allocate_output_index(OutputItemType::FunctionCall);

                                                            
                    let item = json!({
                        "id": item_id,
                        "type": "function_tool_call",
                        "call_id": tool_call.call_id,
                        "name": tool_call.name,
                        "status": "in_progress",
                        "arguments": ""
                    });

                                             
                    let event = emitter.emit_output_item_added(output_index, &item);
                    emitter.send_event(&event, &tx)?;

                                                         
                    let event = emitter.emit_function_call_arguments_delta(
                        output_index,
                        &item_id,
                        &tool_call.arguments,
                    );
                    emitter.send_event(&event, &tx)?;

                                                        
                    let event = emitter.emit_function_call_arguments_done(
                        output_index,
                        &item_id,
                        &tool_call.arguments,
                    );
                    emitter.send_event(&event, &tx)?;

                                          
                    let item_complete = json!({
                        "id": item_id,
                        "type": "function_tool_call",
                        "call_id": tool_call.call_id,
                        "name": tool_call.name,
                        "status": "completed",
                        "arguments": tool_call.arguments
                    });

                                            
                    let event = emitter.emit_output_item_done(output_index, &item_complete);
                    emitter.send_event(&event, &tx)?;

                    emitter.complete_output_item(output_index);
                }

                                                          
                break;
            }

                                                           
            current_request = build_next_request(&state, &current_request);

            continue;
        }

                                                    
        trace!("No tool calls found, ending streaming MCP loop");

                                      
        let reasoning_content = accumulated_response
            .choices
            .first()
            .and_then(|c| c.message.reasoning_content.clone());

                                         
        if let Some(reasoning) = reasoning_content {
            if !reasoning.is_empty() {
                emitter.emit_reasoning_item(&tx, Some(reasoning))?;
            }
        }

                                                                                                  
                                                                                             

                                              
        let usage_json = accumulated_response.usage.as_ref().map(|u| {
            json!({
                "input_tokens": u.prompt_tokens,
                "output_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens
            })
        });
        let event = emitter.emit_completed(usage_json.as_ref());
        emitter.send_event(&event, &tx)?;

        break;
    }

    Ok(())
}

/// Convert chat stream to Responses API events while accumulating for tool call detection
async fn convert_and_accumulate_stream(
    body: Body,
    emitter: &mut ResponseStreamEventEmitter,
    tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
) -> Result<ChatCompletionResponse, String> {
    let mut accumulator = ChatResponseAccumulator::new();
    let mut stream = body.into_data_stream();

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| format!("Stream read error: {}", e))?;

                      
        let event_str = String::from_utf8_lossy(&chunk);
        let event = event_str.trim();

        if event == "data: [DONE]" {
            break;
        }

        if let Some(json_str) = event.strip_prefix("data: ") {
            let json_str = json_str.trim();
            if let Ok(chat_chunk) = serde_json::from_str::<ChatCompletionStreamResponse>(json_str) {
                                                                      
                emitter.process_chunk(&chat_chunk, tx)?;

                                                     
                accumulator.process_chunk(&chat_chunk);
            }
        }
    }

    Ok(accumulator.finalize())
}

/// Accumulates chat streaming chunks into complete ChatCompletionResponse
struct ChatResponseAccumulator {
    id: String,
    model: String,
    content: String,
    reasoning_content: Option<String>,
    tool_calls: HashMap<usize, ToolCall>,
    finish_reason: Option<String>,
    usage: Option<Usage>,
}

impl ChatResponseAccumulator {
    fn new() -> Self {
        Self {
            id: String::new(),
            model: String::new(),
            content: String::new(),
            reasoning_content: None,
            tool_calls: HashMap::new(),
            finish_reason: None,
            usage: None,
        }
    }

    fn process_chunk(&mut self, chunk: &ChatCompletionStreamResponse) {
        if !chunk.id.is_empty() {
            self.id = chunk.id.clone();
        }
        if !chunk.model.is_empty() {
            self.model = chunk.model.clone();
        }

        if let Some(choice) = chunk.choices.first() {
                                 
            if let Some(content) = &choice.delta.content {
                self.content.push_str(content);
            }

                                           
            if let Some(reasoning) = &choice.delta.reasoning_content {
                self.reasoning_content
                    .get_or_insert_with(String::new)
                    .push_str(reasoning);
            }

                                    
            if let Some(tool_call_deltas) = &choice.delta.tool_calls {
                for delta in tool_call_deltas {
                    let index = delta.index as usize;
                    let entry = self.tool_calls.entry(index).or_insert_with(|| ToolCall {
                        id: String::new(),
                        tool_type: "function".to_string(),
                        function: FunctionCallResponse {
                            name: String::new(),
                            arguments: Some(String::new()),
                        },
                    });

                    if let Some(id) = &delta.id {
                        entry.id = id.clone();
                    }
                    if let Some(function) = &delta.function {
                        if let Some(name) = &function.name {
                            entry.function.name = name.clone();
                        }
                        if let Some(args) = &function.arguments {
                            if let Some(ref mut existing_args) = entry.function.arguments {
                                existing_args.push_str(args);
                            }
                        }
                    }
                }
            }

                                    
            if let Some(reason) = &choice.finish_reason {
                self.finish_reason = Some(reason.clone());
            }
        }

                       
        if let Some(usage) = &chunk.usage {
            self.usage = Some(usage.clone());
        }
    }

    fn finalize(self) -> ChatCompletionResponse {
        let mut tool_calls_vec: Vec<_> = self.tool_calls.into_iter().collect();
        tool_calls_vec.sort_by_key(|(index, _)| *index);
        let tool_calls: Vec<_> = tool_calls_vec.into_iter().map(|(_, call)| call).collect();

        ChatCompletionResponse::builder(&self.id, &self.model)
            .choices(vec![ChatChoice {
                index: 0,
                message: ChatCompletionMessage {
                    role: "assistant".to_string(),
                    content: if self.content.is_empty() {
                        None
                    } else {
                        Some(self.content)
                    },
                    tool_calls: if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    },
                    reasoning_content: self.reasoning_content,
                },
                finish_reason: self.finish_reason,
                logprobs: None,
                matched_stop: None,
                hidden_states: None,
            }])
            .maybe_usage(self.usage)
            .build()
    }
}

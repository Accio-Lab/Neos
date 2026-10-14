//! Shared utilities for gRPC routers

use std::{collections::HashMap, sync::Arc};

use axum::response::Response;
use http::StatusCode;
use serde_json::{json, Map, Value};
use smg_grpc_client::sglang_proto::{InputLogProbs, OutputLogProbs};
use tracing::{error, warn};
use uuid::Uuid;

use super::{
    client::GrpcClient,
    context::RequestContext,
    proto_wrapper::{ProtoGenerateComplete, ProtoStream},
    ProcessedMessages,
};
use crate::{
    core::Worker,
    observability::metrics::metrics_labels,
    protocols::{
        chat::{ChatCompletionRequest, ChatMessage},
        common::{
            ChatLogProbs, ChatLogProbsContent, FunctionCallResponse, StringOrArray, Tool, ToolCall,
            ToolChoice, ToolChoiceValue, TopLogProb,
        },
        generate::GenerateFinishReason,
    },
    reasoning_parser::{
        ParserFactory as ReasoningParserFactory, PooledParser as ReasoningPooledParser,
        ReasoningParser,
    },
    routers::{error, grpc::proto_wrapper::ProtoResponseVariant},
    tokenizer::{
        cache::CachedTokenizer,
        chat_template::{ChatTemplateContentFormat, ChatTemplateParams},
        stop::StopSequenceDecoderBuilder,
        traits::Tokenizer,
        HuggingFaceTokenizer, StopSequenceDecoder,
    },
    tool_parser::{
        ParserFactory as ToolParserFactory, PooledParser as ToolPooledParser, ToolParser,
    },
};

/// Resolve tokenizer from registry and cache it in request context.
///
/// This is a helper to avoid duplicating tokenizer resolution logic across
/// preparation stages (chat, generate, embedding).
///
/// Returns the tokenizer Arc, which is also cached in `ctx.state.tokenizer`.
pub(crate) fn resolve_tokenizer(
    ctx: &mut RequestContext,
    stage_name: &str,
) -> Result<Arc<dyn Tokenizer>, Box<Response>> {
    let model_id = ctx.input.model_id.as_deref().ok_or_else(|| {
        error!(
            function = %stage_name,
            "model_id not set in request context"
        );
        Box::new(error::internal_error(
            "model_id_not_set",
            "model_id not set in request context - this is a bug in request routing",
        ))
    })?;

    let tokenizer = ctx
        .components
        .tokenizer_registry
        .get(model_id)
        .ok_or_else(|| {
            error!(
                function = %stage_name,
                model = %model_id,
                "Tokenizer not found for model"
            );
            Box::new(error::internal_error(
                "tokenizer_not_found",
                format!("Tokenizer not found for model: {}", model_id),
            ))
        })?;

                                                                        
    ctx.state.tokenizer = Some(tokenizer.clone());

    Ok(tokenizer)
}

/// Get gRPC client from worker, returning appropriate error response on failure
pub(crate) async fn get_grpc_client_from_worker(
    worker: &Arc<dyn Worker>,
) -> Result<GrpcClient, Response> {
                                                                      
    let client_arc = worker
        .get_grpc_client()
        .await
        .map_err(|e| {
            error!(
                function = "get_grpc_client_from_worker",
                error = %e,
                "Failed to get gRPC client from worker"
            );
            error::internal_error(
                "get_grpc_client_failed",
                format!("Failed to get gRPC client: {}", e),
            )
        })?
        .ok_or_else(|| {
            error!(
                function = "get_grpc_client_from_worker",
                "Selected worker not configured for gRPC"
            );
            error::internal_error(
                "worker_not_configured_for_grpc",
                "Selected worker is not configured for gRPC",
            )
        })?;

    Ok((*client_arc).clone())
}

/// Process tool call arguments in messages
/// Per Transformers docs, tool call arguments in assistant messages should be dicts
fn process_tool_call_arguments(messages: &mut [Value]) -> Result<(), String> {
    for msg in messages {
        let role = msg.get("role").and_then(|v| v.as_str());
        if role != Some("assistant") {
            continue;
        }

        let Some(tool_calls) = msg.get_mut("tool_calls").and_then(|tc| tc.as_array_mut()) else {
            continue;
        };

        for call in tool_calls {
            let Some(function) = call.get_mut("function") else {
                continue;
            };
            let Some(args) = function.get_mut("arguments") else {
                continue;
            };
            let Some(args_str) = args.as_str() else {
                continue;
            };

                                                                   
            match serde_json::from_str::<Value>(args_str) {
                Ok(parsed) => *args = parsed,
                Err(e) => {
                    return Err(format!(
                        "Failed to parse tool call arguments as JSON: '{}'. Error: {}",
                        args_str, e
                    ))
                }
            }
        }
    }
    Ok(())
}

/// Process messages based on content format for ANY message type
pub(crate) fn process_content_format(
    messages: &[ChatMessage],
    content_format: ChatTemplateContentFormat,
) -> Result<Vec<Value>, String> {
    messages
        .iter()
        .map(|message| {
            let mut message_json = serde_json::to_value(message)
                .map_err(|e| format!("Failed to serialize message: {}", e))?;

            if let Some(obj) = message_json.as_object_mut() {
                if let Some(content_value) = obj.get_mut("content") {
                    transform_content_field(content_value, content_format);
                }
            }

            Ok(message_json)
        })
        .collect()
}

/// Transform a single content field based on content format
fn transform_content_field(content_value: &mut Value, content_format: ChatTemplateContentFormat) {
    let Some(content_array) = content_value.as_array() else {
        return;                              
    };

    match content_format {
        ChatTemplateContentFormat::String => {
                                               
            let text_parts: Vec<String> = content_array
                .iter()
                .filter_map(|part| {
                    part.as_object()?
                        .get("type")?
                        .as_str()
                        .filter(|&t| t == "text")
                        .and_then(|_| part.as_object()?.get("text")?.as_str())
                        .map(String::from)
                })
                .collect();

            if !text_parts.is_empty() {
                *content_value = Value::String(text_parts.join(" "));
            }
        }
        ChatTemplateContentFormat::OpenAI => {
                                                               
            let processed_parts: Vec<Value> = content_array
                .iter()
                .map(|part| {
                    part.as_object()
                        .and_then(|obj| obj.get("type")?.as_str())
                        .and_then(|type_str| match type_str {
                            "image_url" => Some(json!({"type": "image"})),
                            "video_url" => Some(json!({"type": "video"})),
                            "audio_url" => Some(json!({"type": "audio"})),
                            _ => None,
                        })
                        .unwrap_or_else(|| part.clone())
                })
                .collect();

            *content_value = Value::Array(processed_parts);
        }
    }
}

/// Generate tool constraints for structured generation
/// Note: tools should already be filtered if needed (by allowed_tools or specific function)
pub(crate) fn generate_tool_constraints(
    tools: &[Tool],
    tool_choice: &Option<ToolChoice>,
    _model: &str,
) -> Result<Option<(String, String)>, String> {
    let Some(choice) = tool_choice.as_ref() else {
        return Ok(None);
    };

    match choice {
                                                               
                                                                                 
        ToolChoice::Function { .. } => {
            if tools.is_empty() {
                return Ok(None);
            }
            let tool = &tools[0];

                                                                                  
            let params_schema = serde_json::to_string(&tool.function.parameters)
                .map_err(|e| format!("Failed to serialize tool parameters: {}", e))?;
            Ok(Some((String::from("json_schema"), params_schema)))
        }

                                                         
        ToolChoice::Value(ToolChoiceValue::Required) => {
            let schema = build_required_array_schema(tools)?;
            Ok(Some(("json_schema".to_string(), schema)))
        }

                                                                      
        ToolChoice::AllowedTools { mode, .. } => {
            if mode == "required" {
                if tools.is_empty() {
                    return Ok(None);
                }
                let schema = build_required_array_schema(tools)?;
                Ok(Some(("json_schema".to_string(), schema)))
            } else {
                                                     
                Ok(None)
            }
        }

                                           
        _ => Ok(None),
    }
}

/// Build JSON schema for required tool calls (array with minItems: 1)
/// Includes $defs consolidation from all tools (matching Python's behavior)
fn build_required_array_schema(tools: &[Tool]) -> Result<String, String> {
    let mut any_of_schemas = Vec::with_capacity(tools.len());
    for tool in tools {
        let tool_schema = json!({
            "properties": {
                "name": {
                    "type": "string",
                    "enum": [tool.function.name]
                },
                "parameters": tool.function.parameters
            },
            "required": ["name", "parameters"]
        });
        any_of_schemas.push(tool_schema);
    }

                                                                                 
    let mut all_defs: Map<String, Value> = Map::new();
    for tool in tools {
        if let Value::Object(params) = &tool.function.parameters {
            if let Some(Value::Object(defs)) = params.get("$defs") {
                for (def_name, def_schema) in defs {
                    if let Some(existing) = all_defs.get(def_name) {
                                              
                        if existing != def_schema {
                            let error_msg = format!(
                                "Tool definition '{}' has multiple conflicting schemas, which is not supported",
                                def_name
                            );
                            error!("{}", error_msg);
                            return Err(error_msg);
                        }
                    } else {
                        all_defs.insert(def_name.clone(), def_schema.clone());
                    }
                }
            }
        }
    }

                                  
    let mut array_schema = json!({
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "anyOf": any_of_schemas
        }
    });

                                                               
    if !all_defs.is_empty() {
        if let Value::Object(ref mut schema_obj) = array_schema {
            schema_obj.insert("$defs".to_string(), Value::Object(all_defs));
        }
    }

    serde_json::to_string(&array_schema)
        .map_err(|e| format!("Failed to serialize tool schema: {}", e))
}

/// Filter tools based on tool_choice (generic helper)
///
/// Returns filtered tools if filtering is needed, otherwise returns None.
/// Used by both Chat API and Responses API (Harmony) for constraint generation.
pub(crate) fn filter_tools_by_tool_choice(
    tools: &[Tool],
    tool_choice: &Option<ToolChoice>,
) -> Option<Vec<Tool>> {
    match tool_choice {
        Some(ToolChoice::AllowedTools { tools: allowed, .. }) => {
            let allowed_names: std::collections::HashSet<&str> =
                allowed.iter().filter_map(|t| t.function_name()).collect();
            let filtered: Vec<Tool> = tools
                .iter()
                .filter(|t| allowed_names.contains(t.function.name.as_str()))
                .cloned()
                .collect();
            Some(filtered)
        }
        Some(ToolChoice::Function { function, .. }) => {
            let filtered: Vec<Tool> = tools
                .iter()
                .filter(|t| t.function.name == function.name)
                .cloned()
                .collect();
            Some(filtered)
        }
        _ => None,                       
    }
}

/// Filter ChatCompletionRequest by tool_choice
///
/// Returns a reference to the original request if no filtering needed,
/// otherwise returns a cloned request with filtered tools.
///
/// Note: Tool existence is validated earlier in ChatCompletionRequest::validate(),
/// so this function assumes tool_choice references valid tools.
pub(crate) fn filter_chat_request_by_tool_choice(
    body: &ChatCompletionRequest,
) -> std::borrow::Cow<'_, ChatCompletionRequest> {
    if let Some(tools) = &body.tools {
        if let Some(filtered_tools) = filter_tools_by_tool_choice(tools, &body.tool_choice) {
            let mut filtered_body = body.clone();
            filtered_body.tools = Some(filtered_tools);
            return std::borrow::Cow::Owned(filtered_body);
        }
    }

                                                    
    std::borrow::Cow::Borrowed(body)
}

/// Process chat messages and apply template (shared by both routers)
/// Requires HuggingFace tokenizer with chat template support
pub(crate) fn process_chat_messages(
    request: &ChatCompletionRequest,
    tokenizer: &dyn Tokenizer,
) -> Result<ProcessedMessages, String> {
                                                                                    
                                                                      
    let hf_tokenizer = tokenizer
        .as_any()
        .downcast_ref::<HuggingFaceTokenizer>()
        .or_else(|| {
                                                                                        
            tokenizer
                .as_any()
                .downcast_ref::<CachedTokenizer>()
                .and_then(|cached| {
                    cached
                        .inner()
                        .as_any()
                        .downcast_ref::<HuggingFaceTokenizer>()
                })
        });

    let formatted_text = if let Some(hf_tokenizer) = hf_tokenizer {
                                                                
        let content_format = hf_tokenizer.chat_template_content_format();
        let mut transformed_messages = process_content_format(&request.messages, content_format)?;

                                                            
        process_tool_call_arguments(&mut transformed_messages)?;

                                                               
        let tools_json: Option<Vec<Value>> = request
            .tools
            .as_ref()
            .map(|tools| {
                tools
                    .iter()
                    .map(serde_json::to_value)
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()
            .map_err(|e| format!("Failed to serialize tools: {}", e))?;

        let kwargs_capacity = 1 + request.chat_template_kwargs.as_ref().map_or(0, |k| k.len());
        let mut combined_template_kwargs = HashMap::with_capacity(kwargs_capacity);

                                                             
        if let Some(reasoning_effort) = &request.reasoning_effort {
            combined_template_kwargs.insert(
                "reasoning_effort".to_string(),
                Value::String(reasoning_effort.clone()),
            );
        }

                                                          
        if let Some(template_kwargs) = &request.chat_template_kwargs {
            for (key, value) in template_kwargs {
                combined_template_kwargs.insert(key.clone(), value.clone());
            }
        }

        let final_template_kwargs = if combined_template_kwargs.is_empty() {
            None
        } else {
            Some(&combined_template_kwargs)
        };

        let params = ChatTemplateParams {
            add_generation_prompt: true,
            tools: tools_json.as_deref(),
            template_kwargs: final_template_kwargs,
            ..Default::default()
        };

                                                             
        let assistant_prefix = if request.continue_final_message
            && !transformed_messages.is_empty()
            && transformed_messages
                .last()
                .and_then(|msg| msg.get("role"))
                .and_then(|v| v.as_str())
                == Some("assistant")
        {
                                                           
            let last_msg = transformed_messages.pop().unwrap();
            last_msg
                .get("content")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        } else {
            None
        };

                                                                               
        let rendered = hf_tokenizer
            .apply_chat_template(&transformed_messages, params)
            .map_err(|e| format!("Failed to apply chat template: {}", e))?;

                                                 
        if let Some(prefix) = assistant_prefix {
            format!("{}{}", rendered, prefix)
        } else {
            rendered
        }
    } else {
        return Err(
            "gRPC router requires HuggingFace tokenizer with chat template support".to_string(),
        );
    };

                                        
    let multimodal_inputs = None;

    Ok(ProcessedMessages {
        text: formatted_text,
        multimodal_inputs,
        stop_sequences: request.stop.clone(),
    })
}

/// Create a StopSequenceDecoder from stop parameters
pub(crate) fn create_stop_decoder(
    tokenizer: &Arc<dyn Tokenizer>,
    stop: Option<&StringOrArray>,
    stop_token_ids: Option<&Vec<u32>>,
    skip_special_tokens: bool,
    no_stop_trim: bool,
) -> StopSequenceDecoder {
                             
    let stop_sequences: Vec<String> = match stop {
        Some(StringOrArray::String(s)) => vec![s.clone()],
        Some(StringOrArray::Array(arr)) => arr.clone(),
        None => vec![],
    };

                                  
    let mut builder =
        StopSequenceDecoderBuilder::new(tokenizer.clone()).skip_special_tokens(skip_special_tokens);

                                                                             
    for seq in stop_sequences {
        builder = if no_stop_trim {
            builder.visible_stop_sequence(seq)
        } else {
            builder.stop_sequence(seq)
        };
    }

                                                                             
    if let Some(token_ids) = stop_token_ids {
        for &token_id in token_ids {
            builder = if no_stop_trim {
                builder.visible_stop_token(token_id)
            } else {
                builder.stop_token(token_id)
            };
        }
    }

    builder.build()
}

/// Parse tool calls from JSON schema constrained response
pub(crate) fn parse_json_schema_response(
    processed_text: &str,
    tool_choice: &Option<ToolChoice>,
    model: &str,
    history_tool_calls_count: usize,
) -> (Option<Vec<ToolCall>>, String) {
    match tool_choice {
        Some(ToolChoice::Function { function, .. }) => {
                                                           
            match serde_json::from_str::<Value>(processed_text) {
                Ok(params) => {
                    let tool_call = ToolCall {
                        id: generate_tool_call_id(
                            model,
                            &function.name,
                            0,
                            history_tool_calls_count,
                        ),
                        tool_type: "function".to_string(),
                        function: FunctionCallResponse {
                            name: function.name.clone(),
                            arguments: Some(
                                serde_json::to_string(&params).unwrap_or_else(|_| "{}".to_string()),
                            ),
                        },
                    };
                    (Some(vec![tool_call]), String::new())
                }
                Err(e) => {
                    error!("Failed to parse specific function parameters: {}", e);
                    (None, processed_text.to_string())
                }
            }
        }
        Some(ToolChoice::Value(ToolChoiceValue::Required))
        | Some(ToolChoice::AllowedTools { .. }) => {
                                                       
            match serde_json::from_str::<Vec<Value>>(processed_text) {
                Ok(parsed_array) => {
                    let spec_tool_calls: Vec<ToolCall> = parsed_array
                        .into_iter()
                        .enumerate()
                        .filter_map(|(i, item)| {
                            let obj = item.as_object()?;
                            let name = obj.get("name")?.as_str()?.to_string();
                            let parameters = obj.get("parameters")?;

                            Some(ToolCall {
                                id: generate_tool_call_id(
                                    model,
                                    &name,
                                    i,
                                    history_tool_calls_count,
                                ),
                                tool_type: "function".to_string(),
                                function: FunctionCallResponse {
                                    name,
                                    arguments: Some(
                                        serde_json::to_string(parameters)
                                            .unwrap_or_else(|_| "{}".to_string()),
                                    ),
                                },
                            })
                        })
                        .collect();
                    (Some(spec_tool_calls), String::new())
                }
                Err(e) => {
                    error!("Failed to parse required tool call array: {}", e);
                    (None, processed_text.to_string())
                }
            }
        }
        _ => (None, processed_text.to_string()),
    }
}

/// Collect responses from a gRPC stream
///
/// This helper processes a gRPC GenerateResponse stream and collects all Complete responses.
/// Used by both regular and PD routers for non-streaming requests.
///
/// # Arguments
/// * `stream` - The gRPC response stream to consume
/// * `worker_name` - Name for logging (e.g., "Prefill", "Decode", "Worker")
///
/// # Returns
/// * `Ok(Vec<GenerateComplete>)` - All complete responses collected from the stream
/// * `Err(Response)` - Error response if the stream fails or returns an error
pub(crate) async fn collect_stream_responses(
    stream: &mut ProtoStream,
    worker_name: &str,
) -> Result<Vec<ProtoGenerateComplete>, Response> {
    let mut all_responses = Vec::new();

    while let Some(response) = stream.next().await {
        match response {
            Ok(gen_response) => {
                match gen_response.into_response() {
                    ProtoResponseVariant::Complete(complete) => {
                        all_responses.push(complete);
                    }
                    ProtoResponseVariant::Error(err) => {
                        error!(function = "collect_stream_responses", worker = %worker_name, error = %err.message(), "Worker generation error");
                                                                                        
                        return Err(error::internal_error(
                            "worker_generation_failed",
                            format!("{} generation failed: {}", worker_name, err.message()),
                        ));
                    }
                    ProtoResponseVariant::Chunk(_chunk) => {
                                                             
                    }
                    ProtoResponseVariant::None => {
                                                            
                    }
                }
            }
            Err(e) => {
                error!(function = "collect_stream_responses", worker = %worker_name, error = ?e, "Worker stream error");
                                                                                
                return Err(error::internal_error(
                    "worker_stream_failed",
                    format!("{} stream failed: {}", worker_name, e),
                ));
            }
        }
    }

    Ok(all_responses)
}

/// Count the number of tool calls in the request message history
/// This is used for KimiK2 format which needs globally unique indices
pub(crate) fn get_history_tool_calls_count(request: &ChatCompletionRequest) -> usize {
    request
        .messages
        .iter()
        .filter_map(|msg| {
            if let ChatMessage::Assistant { tool_calls, .. } = msg {
                tool_calls.as_ref().map(|calls| calls.len())
            } else {
                None
            }
        })
        .sum()
}

/// Generate a tool call ID based on model format
///
/// # Arguments
/// * `model` - Model name to determine ID format
/// * `tool_name` - Name of the tool being called
/// * `tool_index` - Index of this tool call within the current message
/// * `history_count` - Number of tool calls in previous messages
///
/// # Returns
/// A unique ID string. KimiK2 uses `functions.{name}:{global_index}`, others use `call_{uuid}`
pub(crate) fn generate_tool_call_id(
    model: &str,
    tool_name: &str,
    tool_index: usize,
    history_count: usize,
) -> String {
                                                                              
    let is_kimi = model
        .as_bytes()
        .windows(4)                
        .any(|window| window.eq_ignore_ascii_case(b"kimi"));

    if is_kimi {
                                                         
        format!("functions.{}:{}", tool_name, history_count + tool_index)
    } else {
                                                      
        format!("call_{}", &Uuid::new_v4().simple().to_string()[..24])
    }
}

/// Check if a reasoning parser is available for the given model
pub(crate) fn check_reasoning_parser_availability(
    reasoning_parser_factory: &ReasoningParserFactory,
    configured_parser: Option<&str>,
    model: &str,
) -> bool {
    if let Some(parser_name) = configured_parser {
        reasoning_parser_factory.registry().has_parser(parser_name)
    } else {
        reasoning_parser_factory
            .registry()
            .has_parser_for_model(model)
    }
}

/// Check if a tool parser is available for the given model
pub(crate) fn check_tool_parser_availability(
    tool_parser_factory: &ToolParserFactory,
    configured_parser: Option<&str>,
    model: &str,
) -> bool {
    if let Some(parser_name) = configured_parser {
        tool_parser_factory.registry().has_parser(parser_name)
    } else {
        tool_parser_factory.registry().has_parser_for_model(model)
    }
}

/// Get the appropriate reasoning parser for a model
///
/// If a parser name is explicitly configured, use that parser.
/// Otherwise, auto-detect based on the model name.
/// Get a pooled reasoning parser (for non-streaming where state doesn't matter)
pub(crate) fn get_reasoning_parser(
    reasoning_parser_factory: &ReasoningParserFactory,
    configured_parser: Option<&str>,
    model: &str,
) -> ReasoningPooledParser {
    if let Some(parser_name) = configured_parser {
                                             
        reasoning_parser_factory
            .registry()
            .get_pooled_parser(parser_name)
            .unwrap_or_else(|| {
                warn!(
                    "Configured reasoning parser '{}' not found, falling back to model-based selection",
                    parser_name
                );
                reasoning_parser_factory.get_pooled(model)
            })
    } else {
                                     
        reasoning_parser_factory.get_pooled(model)
    }
}

/// Create a fresh reasoning parser instance (for streaming where state isolation is needed)
pub(crate) fn create_reasoning_parser(
    reasoning_parser_factory: &ReasoningParserFactory,
    configured_parser: Option<&str>,
    model: &str,
) -> Option<Box<dyn ReasoningParser>> {
    if let Some(parser_name) = configured_parser {
                                             
        reasoning_parser_factory
            .registry()
            .create_parser(parser_name)
            .or_else(|| {
                warn!(
                    "Configured reasoning parser '{}' not found, falling back to model-based selection",
                    parser_name
                );
                reasoning_parser_factory.registry().create_for_model(model)
            })
    } else {
                                     
        reasoning_parser_factory.registry().create_for_model(model)
    }
}

/// Get the appropriate tool parser for a model
///
/// If a parser name is explicitly configured, use that parser.
/// Otherwise, auto-detect based on the model name.
/// Get a pooled tool parser (for non-streaming where state doesn't matter)
pub(crate) fn get_tool_parser(
    tool_parser_factory: &ToolParserFactory,
    configured_parser: Option<&str>,
    model: &str,
) -> ToolPooledParser {
    if let Some(parser_name) = configured_parser {
                                             
        tool_parser_factory
            .registry()
            .get_pooled_parser(parser_name)
            .unwrap_or_else(|| {
                warn!(
                    "Configured tool parser '{}' not found, falling back to model-based selection",
                    parser_name
                );
                tool_parser_factory.get_pooled(model)
            })
    } else {
                                     
        tool_parser_factory.get_pooled(model)
    }
}

/// Create a fresh tool parser instance (for streaming where state isolation is needed)
pub(crate) fn create_tool_parser(
    tool_parser_factory: &ToolParserFactory,
    configured_parser: Option<&str>,
    model: &str,
) -> Option<Box<dyn ToolParser>> {
    if let Some(parser_name) = configured_parser {
                                             
        tool_parser_factory
            .registry()
            .create_parser(parser_name)
            .or_else(|| {
                warn!(
                    "Configured tool parser '{}' not found, falling back to model-based selection",
                    parser_name
                );
                tool_parser_factory.registry().create_for_model(model)
            })
    } else {
                                     
        tool_parser_factory.registry().create_for_model(model)
    }
}

/// Convert OutputLogProbs to OpenAI ChatLogProbs format
///
/// This function decodes token IDs using the tokenizer and builds the logprobs structure
/// expected by the OpenAI API format.
pub(crate) fn convert_proto_to_openai_logprobs(
    proto_logprobs: &OutputLogProbs,
    tokenizer: &Arc<dyn Tokenizer>,
) -> Result<ChatLogProbs, String> {
    let mut content_items = Vec::with_capacity(proto_logprobs.token_logprobs.len());

                                                                                    
    let token_texts: Vec<String> = proto_logprobs
        .token_ids
        .iter()
        .map(|&token_id| {
            tokenizer
                .decode(&[token_id as u32], false)
                .unwrap_or_else(|_| format!("<token_{}>", token_id))
        })
        .collect();

                                                                                  
    for (i, (&logprob, token_text)) in proto_logprobs
        .token_logprobs
        .iter()
        .zip(token_texts.into_iter())
        .enumerate()
    {
        let bytes = Some(token_text.as_bytes().to_vec());

                                               
        let top_logprobs = if let Some(top_logprobs_entry) = proto_logprobs.top_logprobs.get(i) {
            let mut top_logprobs = Vec::with_capacity(top_logprobs_entry.values.len());

                                                                           
            let top_token_texts: Vec<String> = top_logprobs_entry
                .token_ids
                .iter()
                .map(|&tid| {
                    tokenizer
                        .decode(&[tid as u32], false)
                        .unwrap_or_else(|_| format!("<token_{}>", tid))
                })
                .collect();

            for (j, (&top_logprob, &_top_token_id)) in top_logprobs_entry
                .values
                .iter()
                .zip(top_logprobs_entry.token_ids.iter())
                .enumerate()
            {
                if let Some(top_token_text) = top_token_texts.get(j) {
                    top_logprobs.push(TopLogProb {
                        token: top_token_text.clone(),
                        logprob: top_logprob,
                        bytes: Some(top_token_text.as_bytes().to_vec()),
                    });
                }
            }
            top_logprobs
        } else {
            Vec::new()
        };

        content_items.push(ChatLogProbsContent {
            token: token_text,
            logprob,
            bytes,
            top_logprobs,
        });
    }

    Ok(ChatLogProbs::Detailed {
        content: (!content_items.is_empty()).then_some(content_items),
    })
}

/// Convert OutputLogProbs to Generate format Vec<Vec<Option<f64>>>
///
/// Generate format: [[logprob, token_id, ...], [logprob, token_id, ...], ...]
/// Each inner vec contains [logprob (f64), token_id (i32), ...]
pub(crate) fn convert_generate_output_logprobs(
    proto_logprobs: &OutputLogProbs,
) -> Vec<Vec<Option<f64>>> {
    proto_logprobs
        .token_logprobs
        .iter()
        .zip(proto_logprobs.token_ids.iter())
        .map(|(&logprob, &token_id)| vec![Some(logprob as f64), Some(token_id as f64)])
        .collect()
}

/// Convert InputLogProbs to Generate format Vec<Vec<Option<f64>>>
///
/// Generate format: [[logprob, token_id, ...], [logprob, token_id, ...], ...]
/// First token has null logprob: [[null, token_id], [logprob, token_id], ...]
pub(crate) fn convert_generate_input_logprobs(
    proto_logprobs: &InputLogProbs,
) -> Vec<Vec<Option<f64>>> {
    proto_logprobs
        .token_logprobs
        .iter()
        .zip(proto_logprobs.token_ids.iter())
        .map(|(token_logprob, &token_id)| {
                                                         
            let logprob_value = token_logprob.value.map(|v| v as f64);
            vec![logprob_value, Some(token_id as f64)]
        })
        .collect()
}

/// Parse finish_reason string into GenerateFinishReason enum
///
/// Uses serde to deserialize the finish_reason, which handles all tagged variants automatically.
/// The GenerateFinishReason enum is tagged with `#[serde(tag = "type", rename_all = "lowercase")]`,
/// so it expects JSON objects like:
/// - `{"type":"stop"}` -> Stop
/// - `{"type":"length","length":100}` -> Length { length: 100 }
/// - Any other JSON -> Other(...)
///
/// For backward compatibility, also handles simple string "stop" -> Stop
pub(crate) fn parse_finish_reason(
    reason_str: &str,
    completion_tokens: i32,
) -> GenerateFinishReason {
    if reason_str == "stop" {
        return GenerateFinishReason::Stop;
    }

    if reason_str == "length" {
        return GenerateFinishReason::Length {
            length: completion_tokens.max(0) as u32,
        };
    }

    match serde_json::from_str::<GenerateFinishReason>(reason_str) {
        Ok(finish_reason) => finish_reason,
        Err(_) => match serde_json::from_str::<Value>(reason_str) {
            Ok(json_value) => GenerateFinishReason::Other(json_value),
            Err(_) => GenerateFinishReason::Other(Value::String(reason_str.to_string())),
        },
    }
}

                                                                               
                                                                      
                                                                               

/// Map route path to endpoint label for metrics
pub(crate) fn route_to_endpoint(route: &str) -> &'static str {
    match route {
        "/v1/chat/completions" => metrics_labels::ENDPOINT_CHAT,
        "/generate" => metrics_labels::ENDPOINT_GENERATE,
        "/v1/completions" => metrics_labels::ENDPOINT_COMPLETIONS,
        "/v1/rerank" => metrics_labels::ENDPOINT_RERANK,
        "/v1/responses" => metrics_labels::ENDPOINT_RESPONSES,
        _ => "other",
    }
}

/// Map HTTP status code to error type label for metrics
pub(crate) fn error_type_from_status(status: StatusCode) -> &'static str {
    match status.as_u16() {
        400 => metrics_labels::ERROR_VALIDATION,
        404 => metrics_labels::ERROR_NO_WORKERS,
        408 | 504 => metrics_labels::ERROR_TIMEOUT,
        500..=599 => metrics_labels::ERROR_BACKEND,
        _ => metrics_labels::ERROR_INTERNAL,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{
        protocols::{
            chat::{ChatMessage, MessageContent},
            common::{ContentPart, ImageUrl},
        },
        tokenizer::chat_template::ChatTemplateContentFormat,
    };

    #[test]
    fn test_transform_messages_string_format() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "Hello".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/image.jpg".to_string(),
                        detail: None,
                    },
                },
                ContentPart::Text {
                    text: "World".to_string(),
                },
            ]),
            name: None,
        }];

        let result = process_content_format(&messages, ChatTemplateContentFormat::String).unwrap();

        assert_eq!(result.len(), 1);
        let transformed_message = &result[0];

                                                         
        assert_eq!(
            transformed_message["content"].as_str().unwrap(),
            "Hello World"
        );
        assert_eq!(transformed_message["role"].as_str().unwrap(), "user");
    }

    #[test]
    fn test_transform_messages_openai_format() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "Describe this image:".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/image.jpg".to_string(),
                        detail: Some("high".to_string()),
                    },
                },
            ]),
            name: None,
        }];

        let result = process_content_format(&messages, ChatTemplateContentFormat::OpenAI).unwrap();

        assert_eq!(result.len(), 1);
        let transformed_message = &result[0];

                                                                  
        let content_array = transformed_message["content"].as_array().unwrap();
        assert_eq!(content_array.len(), 2);

                                            
        assert_eq!(content_array[0]["type"], "text");
        assert_eq!(content_array[0]["text"], "Describe this image:");

                                                                     
        assert_eq!(content_array[1], json!({"type": "image"}));
    }

    #[test]
    fn test_transform_messages_simple_string_content() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Text("Simple text message".to_string()),
            name: None,
        }];

        let result = process_content_format(&messages, ChatTemplateContentFormat::String).unwrap();

        assert_eq!(result.len(), 1);
        let transformed_message = &result[0];

                                                        
        assert_eq!(
            transformed_message["content"].as_str().unwrap(),
            "Simple text message"
        );
    }

    #[test]
    fn test_transform_messages_multiple_messages() {
        let messages = vec![
            ChatMessage::System {
                content: MessageContent::Text("System prompt".to_string()),
                name: None,
            },
            ChatMessage::User {
                content: MessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "User message".to_string(),
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "https://example.com/image.jpg".to_string(),
                            detail: None,
                        },
                    },
                ]),
                name: None,
            },
        ];

        let result = process_content_format(&messages, ChatTemplateContentFormat::String).unwrap();

        assert_eq!(result.len(), 2);

                                                 
        assert_eq!(result[0]["role"].as_str().unwrap(), "system");
        assert_eq!(result[0]["content"].as_str().unwrap(), "System prompt");

                                                        
        assert_eq!(result[1]["role"].as_str().unwrap(), "user");
        assert_eq!(result[1]["content"].as_str().unwrap(), "User message");
    }

    #[test]
    fn test_transform_messages_empty_text_parts() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.com/image.jpg".to_string(),
                    detail: None,
                },
            }]),
            name: None,
        }];

        let result = process_content_format(&messages, ChatTemplateContentFormat::String).unwrap();

        assert_eq!(result.len(), 1);
        let transformed_message = &result[0];

                                                                           
        assert!(transformed_message["content"].is_array());
    }

    #[test]
    fn test_transform_messages_mixed_content_types() {
        let messages = vec![
            ChatMessage::User {
                content: MessageContent::Text("Plain text".to_string()),
                name: None,
            },
            ChatMessage::User {
                content: MessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "With image".to_string(),
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "https://example.com/image.jpg".to_string(),
                            detail: Some("low".to_string()),
                        },
                    },
                ]),
                name: None,
            },
        ];

        let result_string =
            process_content_format(&messages, ChatTemplateContentFormat::String).unwrap();

        assert_eq!(result_string.len(), 2);
        assert_eq!(result_string[0]["content"].as_str().unwrap(), "Plain text");
        assert_eq!(result_string[1]["content"].as_str().unwrap(), "With image");

        let result_openai =
            process_content_format(&messages, ChatTemplateContentFormat::OpenAI).unwrap();

        assert_eq!(result_openai.len(), 2);
        assert_eq!(result_openai[0]["content"].as_str().unwrap(), "Plain text");

        let content_array = result_openai[1]["content"].as_array().unwrap();
        assert_eq!(content_array.len(), 2);
        assert_eq!(content_array[0]["type"], "text");
        assert_eq!(content_array[1], json!({"type": "image"}));
    }
}

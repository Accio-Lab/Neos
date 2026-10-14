//! Harmony response processor for non-streaming responses

use std::sync::Arc;

use axum::response::Response;
use smg_grpc_client::sglang_proto::generate_complete::MatchedStop::{
    MatchedStopStr, MatchedTokenId,
};
use tracing::error;

use super::HarmonyParserAdapter;
use crate::{
    protocols::{
        chat::{ChatChoice, ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse},
        common::{CompletionTokensDetails, ToolCall, Usage},
        responses::{
            OutputTokensDetails, ResponseContentPart, ResponseOutputItem, ResponseReasoningContent,
            ResponseStatus, ResponseUsage, ResponsesRequest, ResponsesResponse, ResponsesUsage,
        },
    },
    routers::{
        error,
        grpc::{
            common::{response_collection, response_formatting},
            context::{DispatchMetadata, ExecutionResult},
        },
    },
};

/// Processor for non-streaming Harmony responses
///
/// Collects all output tokens from execution and parses them using
/// HarmonyParserAdapter to extract the complete response.
pub(crate) struct HarmonyResponseProcessor;

impl HarmonyResponseProcessor {
    /// Create a new Harmony response processor
    pub fn new() -> Self {
        Self
    }

    /// Process a non-streaming Harmony chat response
    pub async fn process_non_streaming_chat_response(
        &self,
        execution_result: ExecutionResult,
        chat_request: Arc<ChatCompletionRequest>,
        dispatch: DispatchMetadata,
    ) -> Result<ChatCompletionResponse, Response> {
                                                           
        let all_responses = response_collection::collect_responses(execution_result, false).await?;
        if all_responses.is_empty() {
            return Err(error::internal_error(
                "no_responses_from_server",
                "No responses from server",
            ));
        }

                                                                    
        let mut choices: Vec<ChatChoice> = Vec::new();
        let mut total_reasoning_tokens = 0u32;

        for (index, complete) in all_responses.iter().enumerate() {
                                                      
            let matched_stop = complete.matched_stop().map(|m| match m {
                MatchedTokenId(id) => {
                    serde_json::json!(id)
                }
                MatchedStopStr(s) => {
                    serde_json::json!(s)
                }
            });

                                                               
            let mut parser = HarmonyParserAdapter::new().map_err(|e| {
                error!(
                    function = "process_non_streaming_chat_response",
                    error = %e,
                    "Failed to create Harmony parser"
                );
                error::internal_error(
                    "create_harmony_parser_failed",
                    format!("Failed to create Harmony parser: {}", e),
                )
            })?;

                                                                         
            let parsed = parser
                .parse_complete(
                    complete.output_ids(),
                    complete.finish_reason().to_string(),
                    matched_stop.clone(),
                )
                .map_err(|e| {
                    error!(
                        function = "process_non_streaming_chat_response",
                        error = %e,
                        "Harmony parsing failed on complete response"
                    );
                    error::internal_error(
                        "harmony_parsing_failed",
                        format!("Harmony parsing failed: {}", e),
                    )
                })?;

                                                 
            let message = ChatCompletionMessage {
                role: "assistant".to_string(),
                content: (!parsed.final_text.is_empty()).then_some(parsed.final_text),
                tool_calls: parsed.commentary,
                reasoning_content: parsed.analysis,
            };

            let finish_reason = parsed.finish_reason;

                                                               
            total_reasoning_tokens += parsed.reasoning_token_count;

            choices.push(ChatChoice {
                index: index as u32,
                message,
                logprobs: None,
                finish_reason: Some(finish_reason),
                matched_stop,
                hidden_states: None,
            });
        }

                                        
        let mut usage = response_formatting::build_usage(&all_responses);

                                                                             
        if total_reasoning_tokens > 0 {
            usage.completion_tokens_details = Some(CompletionTokensDetails {
                reasoning_tokens: Some(total_reasoning_tokens),
            });
        }

                                       
        Ok(
            ChatCompletionResponse::builder(&dispatch.request_id, &chat_request.model)
                .created(dispatch.created)
                .choices(choices)
                .usage(usage)
                .maybe_system_fingerprint(dispatch.weight_version.as_deref())
                .build(),
        )
    }
}

impl Default for HarmonyResponseProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of processing a single Responses API iteration
///
/// Used by the MCP tool loop to determine whether to continue
/// executing tools or return the final response.
pub(crate) enum ResponsesIterationResult {
    /// Tool calls found in commentary channel - continue MCP loop
    ToolCallsFound {
        tool_calls: Vec<ToolCall>,
        analysis: Option<String>,                                              
        partial_text: String,                                                
        usage: Usage,                                               
        request_id: String,                                  
    },
    /// No tool calls - return final ResponsesResponse
    Completed {
        response: Box<ResponsesResponse>,
        usage: Usage,
    },
}

impl HarmonyResponseProcessor {
    /// Process a single Responses API iteration
    ///
    /// Parses Harmony channels and determines if tool calls are present.
    /// If tool calls found, returns ToolCallsFound for MCP loop to execute.
    /// If no tool calls, builds final ResponsesResponse.
    ///
    /// # Arguments
    ///
    /// * `execution_result` - The execution result from the model
    /// * `responses_request` - The original Responses API request
    /// * `dispatch` - Dispatch metadata for request tracking
    ///
    /// # Returns
    ///
    /// ResponsesIterationResult indicating whether to continue loop or return
    pub async fn process_responses_iteration(
        &self,
        execution_result: ExecutionResult,
        responses_request: Arc<ResponsesRequest>,
        dispatch: DispatchMetadata,
    ) -> Result<ResponsesIterationResult, Response> {
                                          
        let all_responses = response_collection::collect_responses(execution_result, false).await?;
        if all_responses.is_empty() {
            return Err(error::internal_error(
                "no_responses_from_server",
                "No responses from server",
            ));
        }

                                                                      
        let complete = all_responses
            .first()
            .ok_or_else(|| error::internal_error("no_complete_response", "No complete response"))?;

                                 
        let mut parser = HarmonyParserAdapter::new().map_err(|e| {
            error!(
                function = "process_responses_iteration",
                error = %e,
                "Failed to create Harmony parser"
            );
            error::internal_error(
                "create_harmony_parser_failed",
                format!("Failed to create Harmony parser: {}", e),
            )
        })?;

                                                  
        let matched_stop = complete.matched_stop().map(|m| match m {
            MatchedTokenId(id) => {
                serde_json::json!(id)
            }
            MatchedStopStr(s) => {
                serde_json::json!(s)
            }
        });

        let parsed = parser
            .parse_complete(
                complete.output_ids(),
                complete.finish_reason().to_string(),
                matched_stop,
            )
            .map_err(|e| {
                error!(
                    function = "process_responses_iteration",
                    error = %e,
                    "Harmony parsing failed on complete response"
                );
                error::internal_error(
                    "harmony_parsing_failed",
                    format!("Harmony parsing failed: {}", e),
                )
            })?;

                                                                              
                                                                                
                                                                                 
        let messages = parser.get_messages();
        let tool_messages_generated = messages.iter().any(|msg| {
            msg.author.role == openai_harmony::chat::Role::Tool
                && msg.recipient.as_deref() == Some("assistant")
        });

        if tool_messages_generated {
            tracing::warn!(
                "Model generated Tool->Assistant message instead of Assistant message. \
                This is a model hallucination bug where it copies tool result format."
            );
        }

                                                                     
        let mut usage = response_formatting::build_usage(std::slice::from_ref(complete));

                                                                             
        if parsed.reasoning_token_count > 0 {
            usage.completion_tokens_details = Some(CompletionTokensDetails {
                reasoning_tokens: Some(parsed.reasoning_token_count),
            });
        }

                                                     
        if let Some(tool_calls) = parsed.commentary {
                                                               
            return Ok(ResponsesIterationResult::ToolCallsFound {
                tool_calls,
                analysis: parsed.analysis,
                partial_text: parsed.final_text,
                usage,
                request_id: dispatch.request_id.clone(),
            });
        }

                                                        
        let mut output: Vec<ResponseOutputItem> = Vec::new();

                                                               
        if let Some(analysis) = parsed.analysis {
            let reasoning_item = ResponseOutputItem::Reasoning {
                id: format!("reasoning_{}", dispatch.request_id),
                summary: vec![],
                content: vec![ResponseReasoningContent::ReasoningText { text: analysis }],
                status: Some("completed".to_string()),
            };
            output.push(reasoning_item);
        }

                                                          
        if !parsed.final_text.is_empty() {
            let message_item = ResponseOutputItem::Message {
                id: format!("msg_{}", dispatch.request_id),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text: parsed.final_text,
                    annotations: vec![],
                    logprobs: None,
                }],
                status: "completed".to_string(),
            };
            output.push(message_item);
        }

                                                           
        let response = ResponsesResponse::builder(&dispatch.request_id, &responses_request.model)
            .copy_from_request(&responses_request)
            .created_at(dispatch.created as i64)
            .status(ResponseStatus::Completed)
            .output(output)
            .maybe_text(responses_request.text.clone())
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
            .build();

        Ok(ResponsesIterationResult::Completed {
            response: Box::new(response),
            usage,
        })
    }
}

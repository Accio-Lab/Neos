//! FFI module for exposing sgl-model-gateway preprocessing and postprocessing functions
//! to C-compatible languages (e.g., Golang via cgo)
//!
//! This module provides C-compatible function signatures for:
//! - Tokenizer operations (encode, decode, chat template)
//! - Tool parser operations (parse tool calls)
//! - Tool constraint generation
//! - gRPC client SDK (complete request-response flow)
//!
//! # Safety
//! All functions marked with `#[no_mangle]` and `extern "C"` must be called
//! with valid pointers and follow the documented memory management rules.

                        
pub use error::{SglErrorCode, set_error_message, set_error_message_fmt, clear_error_message};

                                        
pub use memory::{sgl_free_string, sgl_free_token_ids};

                                
pub use tokenizer::{
    TokenizerHandle,
    sgl_tokenizer_create_from_file,
    sgl_tokenizer_encode,
    sgl_tokenizer_apply_chat_template,
    sgl_tokenizer_apply_chat_template_with_tools,
    sgl_tokenizer_decode,
    sgl_tokenizer_free,
};

                                  
pub use tool_parser::{
    ToolParserHandle,
    sgl_tool_parser_create,
    sgl_tool_parser_parse_complete,
    sgl_tool_parser_parse_incremental,
    sgl_tool_parser_reset,
    sgl_tool_parser_free,
};

                                     
pub use grpc_converter::{
    GrpcResponseConverterHandle,
    sgl_grpc_response_converter_create,
    sgl_grpc_response_converter_convert_chunk,
    sgl_grpc_response_converter_free,
};

                                 
pub use client::{
    SglangClientHandle,
    sgl_client_create,
    sgl_client_free,
};

                             
pub use stream::{
    SglangStreamHandle,
    sgl_stream_read_next,
    sgl_stream_free,
};

                                                                             
pub use client::sgl_client_chat_completion_stream;

                                   
pub use preprocessor::{
    sgl_preprocess_chat_request,
    sgl_preprocess_chat_request_with_tokenizer,
    sgl_preprocessed_request_free,
};

                                    
pub use postprocessor::{
    sgl_postprocess_stream_chunk,
    sgl_postprocess_stream_chunks_batch,
};

                              
pub use utils::sgl_generate_tool_constraints;

              
mod error;
mod memory;
mod tokenizer;
mod tool_parser;
mod grpc_converter;
mod client;
mod stream;
mod utils;
mod preprocessor;
mod postprocessor;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes() {
        assert_eq!(SglErrorCode::Success as i32, 0);
        assert_eq!(SglErrorCode::InvalidArgument as i32, 1);
    }
}

//! gRPC router implementations

use smg_grpc_client::sglang_proto::MultimodalInputs;

use crate::protocols::common::StringOrArray;

pub mod client;                 
pub(crate) mod common;
pub(crate) mod context;
pub(crate) mod harmony;
pub(crate) mod pd_router;                           
pub(crate) mod pipeline;
pub(crate) mod proto_wrapper;
pub(crate) mod regular;
pub(crate) mod router;                           
pub(crate) mod utils;                        

/// Processed chat messages ready for gRPC generation
#[derive(Debug)]
pub(crate) struct ProcessedMessages {
    pub text: String,
    pub multimodal_inputs: Option<MultimodalInputs>,
    #[allow(dead_code)]
    pub stop_sequences: Option<StringOrArray>,
}

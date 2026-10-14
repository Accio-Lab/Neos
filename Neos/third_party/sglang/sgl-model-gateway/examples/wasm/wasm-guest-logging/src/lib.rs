//! WASM Guest Logging Example for sgl-model-gateway
//!
//! This example demonstrates logging and tracing middleware
//! for sgl-model-gateway using the WebAssembly Component Model.
//!
//! Features:
//! - Request tracking and tracing headers
//! - Response status code conversion

wit_bindgen::generate!({
    path: "../../../src/wasm/interface",
    world: "sgl-model-gateway",
});

use exports::sgl::model_gateway::{
    middleware_on_request::Guest as OnRequestGuest,
    middleware_on_response::Guest as OnResponseGuest,
};
use sgl::model_gateway::middleware_types::{Action, Header, ModifyAction, Request, Response};

/// Main middleware implementation
struct Middleware;

                                   
fn create_header(name: &str, value: &str) -> Header {
    Header {
        name: name.to_string(),
        value: value.to_string(),
    }
}

                                 
impl OnRequestGuest for Middleware {
    fn on_request(req: Request) -> Action {
        let mut modify_action = ModifyAction {
            status: None,
            headers_set: vec![],
            headers_add: vec![],
            headers_remove: vec![],
            body_replace: None,
        };

                                      
                                              
        modify_action
            .headers_add
            .push(create_header("x-request-id", &req.request_id));
        modify_action
            .headers_add
            .push(create_header("x-wasm-processed", "true"));
        modify_action.headers_add.push(create_header(
            "x-processed-at",
            &req.now_epoch_ms.to_string(),
        ));

                                             
        if req.path.starts_with("/api") || req.path.starts_with("/v1") {
            modify_action
                .headers_add
                .push(create_header("x-api-route", "true"));
        }

        Action::Modify(modify_action)
    }
}

                                  
impl OnResponseGuest for Middleware {
    fn on_response(resp: Response) -> Action {
                                                                                
        if resp.status == 500 {
            let modify_action = ModifyAction {
                status: Some(503),
                headers_set: vec![],
                headers_add: vec![],
                headers_remove: vec![],
                body_replace: None,
            };
            Action::Modify(modify_action)
        } else {
                                     
            Action::Continue
        }
    }
}

// Export the component
export!(Middleware);

                                                                                       
package ffi

/*
#cgo LDFLAGS: -lsgl_model_gateway_go -ldl
#include <stdlib.h>
#include <stdint.h>

// Error codes (must match client.go)
typedef enum {
    SGL_ERROR_SUCCESS = 0,
    SGL_ERROR_INVALID_ARGUMENT = 1,
    SGL_ERROR_TOKENIZATION_ERROR = 2,
    SGL_ERROR_PARSING_ERROR = 3,
    SGL_ERROR_MEMORY_ERROR = 4,
    SGL_ERROR_UNKNOWN = 99
} SglErrorCode;

// Preprocessor functions
SglErrorCode sgl_preprocess_chat_request(
    const char* request_json,
    const char* tokenizer_path,
    char** prompt_text_out,
    uint32_t** token_ids_out,
    size_t* token_ids_len_out,
    char** tool_constraints_json_out,
    int32_t* prompt_tokens_out,
    char** error_out
);

// Opaque handle (must match grpc_converter.go)
typedef void* TokenizerHandle;

SglErrorCode sgl_preprocess_chat_request_with_tokenizer(
    const char* request_json,
    void* tokenizer_handle,
    char** prompt_text_out,
    uint32_t** token_ids_out,
    size_t* token_ids_len_out,
    char** tool_constraints_json_out,
    int32_t* prompt_tokens_out,
    char** error_out
);

void sgl_preprocessed_request_free(
    char* prompt_text,
    uint32_t* token_ids,
    size_t token_ids_len,
    char* tool_constraints_json
);

// Memory management
void sgl_free_string(char* s);
void sgl_free_token_ids(uint32_t* ptr, size_t count);
*/
import "C"

import (
	"fmt"
	"unsafe"
)

                                                             
type PreprocessedRequest struct {
	PromptText          string
	TokenIDs            []uint32
	ToolConstraintsJSON string
	PromptTokens        int32
	                                          
	promptTextPtr          *C.char
	tokenIDsPtr            *C.uint32_t
	tokenIDsLen            uintptr
	toolConstraintsJSONPtr *C.char
}

                                                               
  
                 
                                       
                                  
                                                       
  
                                                       
func PreprocessChatRequest(requestJSON, tokenizerPath string) (*PreprocessedRequest, error) {
	requestJSONC := C.CString(requestJSON)
	defer C.free(unsafe.Pointer(requestJSONC))

	tokenizerPathC := C.CString(tokenizerPath)
	defer C.free(unsafe.Pointer(tokenizerPathC))

	var promptTextOut *C.char
	var tokenIDsOut *C.uint32_t
	var tokenIDsLenOut C.size_t
	var toolConstraintsJSONOut *C.char
	var promptTokensOut C.int32_t
	var errorOut *C.char

	errorCode := C.sgl_preprocess_chat_request(
		requestJSONC,
		tokenizerPathC,
		&promptTextOut,
		&tokenIDsOut,
		&tokenIDsLenOut,
		&toolConstraintsJSONOut,
		&promptTokensOut,
		&errorOut,
	)

	if errorCode != C.SGL_ERROR_SUCCESS {
		errorMsg := ""
		if errorOut != nil {
			errorMsg = C.GoString(errorOut)
			C.sgl_free_string(errorOut)
		}
		return nil, fmt.Errorf("preprocessing failed: %s", errorMsg)
	}

	result := &PreprocessedRequest{
		PromptText:          C.GoString(promptTextOut),
		TokenIDs:            make([]uint32, tokenIDsLenOut),
		ToolConstraintsJSON: "",
		PromptTokens:        int32(promptTokensOut),
	}

	                 
	if tokenIDsOut != nil && tokenIDsLenOut > 0 {
		tokenIDsSlice := (*[1 << 30]C.uint32_t)(unsafe.Pointer(tokenIDsOut))[:tokenIDsLenOut:tokenIDsLenOut]
		for i := range result.TokenIDs {
			result.TokenIDs[i] = uint32(tokenIDsSlice[i])
		}
	}

	                                        
	if toolConstraintsJSONOut != nil {
		result.ToolConstraintsJSON = C.GoString(toolConstraintsJSONOut)
	}

	                                   
	result.promptTextPtr = promptTextOut
	result.tokenIDsPtr = tokenIDsOut
	result.tokenIDsLen = uintptr(tokenIDsLenOut)
	result.toolConstraintsJSONPtr = toolConstraintsJSONOut

	return result, nil
}

                                                                                                               
  
                                                                                   
                                                                                        
                                                                          
  
                                                       
func PreprocessChatRequestWithTokenizer(requestJSON string, tokenizerHandle *TokenizerHandle) (*PreprocessedRequest, error) {
	requestJSONC := C.CString(requestJSON)
	defer C.free(unsafe.Pointer(requestJSONC))

	if tokenizerHandle == nil || tokenizerHandle.handle == nil {
		return nil, fmt.Errorf("invalid tokenizer handle")
	}

	var promptTextOut *C.char
	var tokenIDsOut *C.uint32_t
	var tokenIDsLenOut C.size_t
	var toolConstraintsJSONOut *C.char
	var promptTokensOut C.int32_t
	var errorOut *C.char

	errorCode := C.sgl_preprocess_chat_request_with_tokenizer(
		requestJSONC,
		unsafe.Pointer(tokenizerHandle.handle),                                       
		&promptTextOut,
		&tokenIDsOut,
		&tokenIDsLenOut,
		&toolConstraintsJSONOut,
		&promptTokensOut,
		&errorOut,
	)

	if errorCode != C.SGL_ERROR_SUCCESS {
		errorMsg := ""
		if errorOut != nil {
			errorMsg = C.GoString(errorOut)
			C.sgl_free_string(errorOut)
		}
		return nil, fmt.Errorf("preprocessing failed: %s", errorMsg)
	}

	result := &PreprocessedRequest{
		PromptText:          C.GoString(promptTextOut),
		TokenIDs:            make([]uint32, tokenIDsLenOut),
		ToolConstraintsJSON: "",
		PromptTokens:        int32(promptTokensOut),
	}

	                 
	if tokenIDsOut != nil && tokenIDsLenOut > 0 {
		tokenIDsSlice := (*[1 << 30]C.uint32_t)(unsafe.Pointer(tokenIDsOut))[:tokenIDsLenOut:tokenIDsLenOut]
		for i := range result.TokenIDs {
			result.TokenIDs[i] = uint32(tokenIDsSlice[i])
		}
	}

	                                        
	if toolConstraintsJSONOut != nil {
		result.ToolConstraintsJSON = C.GoString(toolConstraintsJSONOut)
	}

	                                   
	result.promptTextPtr = promptTextOut
	result.tokenIDsPtr = tokenIDsOut
	result.tokenIDsLen = uintptr(tokenIDsLenOut)
	result.toolConstraintsJSONPtr = toolConstraintsJSONOut

	return result, nil
}

                                                             
func (p *PreprocessedRequest) Free() {
	if p.promptTextPtr != nil || p.tokenIDsPtr != nil || p.toolConstraintsJSONPtr != nil {
		C.sgl_preprocessed_request_free(
			p.promptTextPtr,
			p.tokenIDsPtr,
			C.size_t(p.tokenIDsLen),
			p.toolConstraintsJSONPtr,
		)
		                                        
		p.promptTextPtr = nil
		p.tokenIDsPtr = nil
		p.tokenIDsLen = 0
		p.toolConstraintsJSONPtr = nil
	}
}

                                                                                
                                                               
func FreePreprocessedRequest(promptTextPtr *C.char, tokenIDsPtr *C.uint32_t, tokenIDsLen uintptr, toolConstraintsJSONPtr *C.char) {
	if promptTextPtr != nil || tokenIDsPtr != nil || toolConstraintsJSONPtr != nil {
		C.sgl_preprocessed_request_free(
			promptTextPtr,
			tokenIDsPtr,
			C.size_t(tokenIDsLen),
			toolConstraintsJSONPtr,
		)
	}
}

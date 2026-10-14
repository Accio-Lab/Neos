package handlers

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	sglang "github.com/sglang/sglang-go-grpc-sdk"
	"github.com/valyala/fasthttp"
	"go.uber.org/zap"

	"oai_server/models"
	"oai_server/service"
	"oai_server/utils"
)

                                               
type ChatHandler struct {
	logger  *zap.Logger
	service *service.SGLangService
}

                                            
func NewChatHandler(logger *zap.Logger, svc *service.SGLangService) *ChatHandler {
	return &ChatHandler{
		logger:  logger,
		service: svc,
	}
}

                                                   
type recvResult struct {
	chunkJSON string
	err       error
}

                                                         
func (h *ChatHandler) HandleChatCompletion(ctx *fasthttp.RequestCtx) {
	var req models.ChatRequest
	if err := json.Unmarshal(ctx.PostBody(), &req); err != nil {
		h.logger.Warn("Invalid chat completion request", zap.Error(err))
		utils.RespondError(ctx, 400, fmt.Sprintf("Invalid request: %v", err), "invalid_request_error")
		return
	}

	path := string(ctx.Path())

	defer func() {
		statusCode := ctx.Response.StatusCode()
		if statusCode == 0 {
			statusCode = 200
		}
		h.logHTTPResponse(statusCode, path)
	}()

	                           
	messages := make([]sglang.ChatMessage, len(req.Messages))
	for i, msg := range req.Messages {
		role, roleOk := msg["role"]
		content, contentOk := msg["content"]

		                
		if !roleOk || role == "" {
			h.logger.Warn("Missing or empty role in message", zap.Int("message_index", i))
			utils.RespondError(ctx, 400, "Message role is required and cannot be empty", "invalid_request_error")
			return
		}

		                                               
		                                                                    
		                                                  
		contentStr := ""
		if contentOk && content != "" {
			contentStr = content
		}

		messages[i] = sglang.ChatMessage{
			Role:    role,
			Content: contentStr,
		}
	}

	sglReq := sglang.ChatCompletionRequest{
		Model:    req.Model,
		Messages: messages,
		Stream:   req.Stream,
	}

	if req.Temperature != nil {
		temp := float32(*req.Temperature)
		sglReq.Temperature = &temp
	}
	if req.TopP != nil {
		topP := float32(*req.TopP)
		sglReq.TopP = &topP
	}
	if req.MaxCompletionTokens != nil {
		sglReq.MaxCompletionTokens = req.MaxCompletionTokens
	} else if req.MaxTokens != nil {
		sglReq.MaxCompletionTokens = req.MaxTokens
	}

	requestCtx := context.Background()

	if req.Stream {
		h.handleStreamingCompletion(ctx, requestCtx, sglReq)
	} else {
		h.handleNonStreamingCompletion(ctx, requestCtx, sglReq)
	}
}

                                                                                     
func isBrokenPipeError(err error) bool {
	if err == nil {
		return false
	}
	errStr := err.Error()
	return strings.Contains(errStr, "broken pipe") ||
		strings.Contains(errStr, "connection reset by peer") ||
		strings.Contains(errStr, "connection closed") ||
		strings.Contains(errStr, "write: connection closed")
}

                                                         
func (h *ChatHandler) logHTTPResponse(statusCode int, path string) {
	var statusText string
	var colorCode string

	switch {
	case statusCode >= 200 && statusCode < 300:
		colorCode = "\033[32m"         
		statusText = "OK"
	case statusCode >= 300 && statusCode < 400:
		colorCode = "\033[33m"          
		statusText = "Redirect"
	case statusCode >= 400 && statusCode < 500:
		colorCode = "\033[33m"          
		statusText = "Client Error"
	case statusCode >= 500:
		colorCode = "\033[31m"       
		statusText = "Server Error"
	default:
		colorCode = "\033[37m"         
		statusText = "Unknown"
	}

	resetCode := "\033[0m"
	msg := fmt.Sprintf("%s[%d %s]%s %s", colorCode, statusCode, statusText, resetCode, path)
	h.logger.Info(msg)
}

func (h *ChatHandler) handleStreamingCompletion(ctx *fasthttp.RequestCtx, requestCtx context.Context, req sglang.ChatCompletionRequest) {

	ctx.SetContentType("text/event-stream")
	ctx.Response.Header.Set("Cache-Control", "no-cache")
	ctx.Response.Header.Set("Connection", "keep-alive")
	ctx.Response.Header.Set("X-Accel-Buffering", "no")
	ctx.SetStatusCode(200)

	var clientDisconnected bool
	                                                                    
	                                                                                             
	const flushTimeout = 5 * time.Second

	ctx.SetBodyStreamWriter(func(w *bufio.Writer) {
		streamCtx, cancel := context.WithCancel(context.Background())
		defer cancel()

		stream, err := h.service.Client().CreateChatCompletionStream(streamCtx, req)
		if err != nil {
			h.logger.Error("Failed to create chat completion stream",
				zap.Error(err),
				zap.String("model", req.Model),
			)
			                                                      
			errInfo, sendErr := h.sendSSEError(w, err)
			if sendErr != nil {
				h.logger.Warn("Failed to send SSE error", zap.Error(sendErr))
			} else if errInfo.IsTimeout {
				h.logger.Error("Stream creation timeout", zap.Error(err))
			}
			return
		}
		defer func() {
			if closeErr := stream.Close(); closeErr != nil {
				h.logger.Warn("Failed to close stream", zap.Error(closeErr))
			}
		}()

		                                                                                                
		recvChan := make(chan recvResult, 20)
		recvGoroutineDone := make(chan struct{})
		go func() {
			defer func() {
				close(recvChan)
				close(recvGoroutineDone)
			}()
			for {
				                                                                                    
				select {
				case <-streamCtx.Done():
					return
				default:
				}

				                                                                       
				                                                    
				chunkJSON, err := stream.RecvJSON()

				                                               
				select {
				case <-streamCtx.Done():
					return
				default:
				}

				                                                 
				                                                                    
				                                                                  
				select {
				case recvChan <- recvResult{chunkJSON: chunkJSON, err: err}:
					if err != nil {
						                                         
						return
					}
				case <-streamCtx.Done():
					                                                      
					return
				}
			}
		}()

		for {
			if clientDisconnected {
				cancel()
				                                                       
				stream.Close()
				return
			}

			select {
			case <-streamCtx.Done():
				                                                       
				stream.Close()
				return
			case result, ok := <-recvChan:
				if !ok {
					                               
					return
				}
				if result.err == io.EOF {
					if !clientDisconnected {
						w.WriteString("data: [DONE]\n\n")
						                                         
						flushDone := make(chan error, 1)
						go func() {
							flushDone <- w.Flush()
						}()
						flushCtx, flushCancel := context.WithTimeout(streamCtx, flushTimeout)
						defer flushCancel()
						select {
						case flushErr := <-flushDone:
							if flushErr != nil && !isBrokenPipeError(flushErr) {
								h.logger.Warn("Final flush error", zap.Error(flushErr))
							}
						case <-flushCtx.Done():
							if flushCtx.Err() == context.DeadlineExceeded {
								h.logger.Warn("Final flush timeout", zap.Duration("timeout", flushTimeout))
							}
						case <-streamCtx.Done():
							                                
						}
					}
					return
				}
				if result.err != nil {
					if result.err == context.Canceled || result.err == context.DeadlineExceeded {
						return
					}
					                                      
					errInfo, sendErr := h.sendSSEError(w, result.err)
					if sendErr != nil {
						h.logger.Warn("Failed to send SSE error", zap.Error(sendErr))
					}
					if errInfo.IsTimeout {
						h.logger.Error("Stream timeout error", zap.Error(result.err))
					} else {
						h.logger.Error("Stream error", zap.Error(result.err))
					}
					return
				}
				if result.chunkJSON == "" {
					continue
				}

				w.WriteString("data: ")
				w.WriteString(result.chunkJSON)
				w.WriteString("\n\n")

				                                          
				                                                                                   
				                                                  
				                                                                         
				                                                             
				flushDone := make(chan error, 1)
				go func() {
					flushDone <- w.Flush()
				}()

				flushCtx, flushCancel := context.WithTimeout(streamCtx, flushTimeout)
				defer flushCancel()

				select {
				case err := <-flushDone:
					if err != nil {
						if isBrokenPipeError(err) {
							clientDisconnected = true
							cancel()
							                                                       
							stream.Close()
							return
						}
						h.logger.Warn("Flush error", zap.Error(err))
					}
				case <-flushCtx.Done():
					                                                    
					                                                                  
					if flushCtx.Err() == context.DeadlineExceeded {
						h.logger.Warn("Flush timeout, client may be slow or disconnected", zap.Duration("timeout", flushTimeout))
					}
					clientDisconnected = true
					cancel()
					stream.Close()
					return
				case <-streamCtx.Done():
					                                   
					return
				}
			}
		}
	})
}

func (h *ChatHandler) handleNonStreamingCompletion(ctx *fasthttp.RequestCtx, requestCtx context.Context, req sglang.ChatCompletionRequest) {
	resp, err := h.service.Client().CreateChatCompletion(requestCtx, req)
	if err != nil {
		h.logger.Error("Failed to create chat completion",
			zap.Error(err),
			zap.String("model", req.Model),
		)
		utils.RespondError(ctx, 500, fmt.Sprintf("Failed to create completion: %v", err), "server_error")
		return
	}

	                           
	response := utils.BuildResponseBase(resp.ID, resp.Created, resp.Model)
	response["object"] = "chat.completion"

	choices := make([]map[string]interface{}, len(resp.Choices))
	for i, choice := range resp.Choices {
		choiceMap := map[string]interface{}{
			"index": choice.Index,
			"message": map[string]interface{}{
				"role":    choice.Message.Role,
				"content": choice.Message.Content,
			},
			"finish_reason": choice.FinishReason,
		}
		if len(choice.Message.ToolCalls) > 0 {
			toolCalls := make([]map[string]interface{}, len(choice.Message.ToolCalls))
			for j, tc := range choice.Message.ToolCalls {
				toolCalls[j] = map[string]interface{}{
					"id":       tc.ID,
					"type":     tc.Type,
					"function": map[string]interface{}{"name": tc.Function.Name, "arguments": tc.Function.Arguments},
				}
			}
			choiceMap["message"].(map[string]interface{})["tool_calls"] = toolCalls
		}
		choices[i] = choiceMap
	}
	response["choices"] = choices

	                                          
	response["usage"] = map[string]interface{}{
		"prompt_tokens":     resp.Usage.PromptTokens,
		"completion_tokens": resp.Usage.CompletionTokens,
		"total_tokens":      resp.Usage.TotalTokens,
	}

	ctx.SetStatusCode(200)
	ctx.SetContentType("application/json")
	jsonData, _ := json.Marshal(response)
	ctx.Write(jsonData)
}

                                                 
type StreamErrorInfo struct {
	Message   string
	Type      string
	Code      int
	IsTimeout bool
}

                                              
func parseStreamError(err error) StreamErrorInfo {
	if err == nil {
		return StreamErrorInfo{}
	}

	errorMsg := err.Error()
	                                        
	isTimeout := strings.HasPrefix(errorMsg, "stream.Recv() timeout") || strings.Contains(errorMsg, "timeout after")

	errorType := "server_error"
	errorCode := 500
	if isTimeout {
		errorType = "timeout_error"
		errorCode = 504
	}

	return StreamErrorInfo{
		Message:   errorMsg,
		Type:      errorType,
		Code:      errorCode,
		IsTimeout: isTimeout,
	}
}

                                               
func formatErrorJSON(errInfo StreamErrorInfo) string {
	errorObj := map[string]interface{}{
		"error": map[string]interface{}{
			"message": errInfo.Message,
			"type":    errInfo.Type,
			"code":    errInfo.Code,
		},
	}
	jsonBytes, _ := json.Marshal(errorObj)
	return string(jsonBytes)
}

                                                                    
func (h *ChatHandler) sendSSEError(w *bufio.Writer, err error) (StreamErrorInfo, error) {
	errInfo := parseStreamError(err)
	errorJSON := formatErrorJSON(errInfo)

	w.WriteString("data: ")
	w.WriteString(errorJSON)
	w.WriteString("\n\n")

	if flushErr := w.Flush(); flushErr != nil && !isBrokenPipeError(flushErr) {
		h.logger.Warn("Failed to flush error response", zap.Error(flushErr))
		return errInfo, flushErr
	}

	return errInfo, nil
}

                                                            
func (h *ChatHandler) HandleGenerate(ctx *fasthttp.RequestCtx) {
	path := string(ctx.Path())

	defer func() {
		statusCode := ctx.Response.StatusCode()
		if statusCode == 0 {
			statusCode = 200
		}
		h.logHTTPResponse(statusCode, path)
	}()

	                     
	var req map[string]interface{}
	if err := json.Unmarshal(ctx.PostBody(), &req); err != nil {
		h.logger.Warn("Invalid generate request", zap.Error(err))
		utils.RespondError(ctx, 400, fmt.Sprintf("Invalid request: %v", err), "invalid_request_error")
		return
	}

	                                   
	text, ok := req["text"].(string)
	if !ok || text == "" {
		utils.RespondError(ctx, 400, "Missing or invalid 'text' field", "invalid_request_error")
		return
	}

	samplingParams, _ := req["sampling_params"].(map[string]interface{})
	if samplingParams == nil {
		samplingParams = make(map[string]interface{})
	}

	                                                   
	chatReq := sglang.ChatCompletionRequest{
		Model:    "default",
		Messages: []sglang.ChatMessage{{Role: "user", Content: text}},
		Stream:   false,
	}

	                       
	if maxNewTokens, ok := samplingParams["max_new_tokens"].(float64); ok {
		tokens := int(maxNewTokens)
		chatReq.MaxCompletionTokens = &tokens
	}
	if temp, ok := samplingParams["temperature"].(float64); ok {
		temp32 := float32(temp)
		chatReq.Temperature = &temp32
	}
	if topP, ok := samplingParams["top_p"].(float64); ok {
		topP32 := float32(topP)
		chatReq.TopP = &topP32
	}
	if topK, ok := samplingParams["top_k"].(float64); ok {
		topKInt := int(topK)
		chatReq.TopK = &topKInt
	}

	requestCtx := context.Background()

	                                                      
	resp, err := h.service.Client().CreateChatCompletion(requestCtx, chatReq)
	if err != nil {
		h.logger.Error("Failed to create completion",
			zap.Error(err),
		)
		utils.RespondError(ctx, 500, fmt.Sprintf("Failed to create completion: %v", err), "server_error")
		return
	}

	                                              
	                                                                                    
	finishReason := resp.Choices[0].FinishReason
	if finishReason == "" {
		finishReason = "stop"
	}

	response := map[string]interface{}{
		"text": resp.Choices[0].Message.Content,
		"meta_info": map[string]interface{}{
			"id":                resp.ID,
			"finish_reason":     finishReason,
			"prompt_tokens":     resp.Usage.PromptTokens,
			"completion_tokens": resp.Usage.CompletionTokens,
			"cached_tokens":     0,                                           
			"weight_version":    "",                                          
		},
	}

	ctx.SetStatusCode(200)
	ctx.SetContentType("application/json")
	jsonData, _ := json.Marshal(response)
	ctx.Write(jsonData)
}

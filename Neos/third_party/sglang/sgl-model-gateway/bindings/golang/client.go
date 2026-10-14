                                                        
  
                                                                                       
                                                                                          
  
               
  
                                                       
                                            
                                        
     
                  
                  
    
                       
  
                                                                              
                     
                                   
                                      
      
     
  
                                                                   
package sglang

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"

	grpcclient "github.com/sglang/sglang-go-grpc-sdk/internal/grpc"
)

                                                                  
                                                                            
                                      
  
                                                               
type Client struct {
	endpoint      string
	tokenizerPath string
	grpcClient    *grpcclient.GrpcClient                     
	mu            sync.RWMutex
}

                                                              
type ClientConfig struct {
	                                                                      
	                                                                     
	Endpoint string

	                                                                  
	                                                                    
	                  
	TokenizerPath string

	                                                                    
	                                                                        
	ChannelBufferSizes *ChannelBufferSizes

	                                                             
	                                       
	Timeouts *Timeouts
}

                                                                    
                                                                       
                                             
type ChannelBufferSizes = grpcclient.ChannelBufferSizes

                                                             
type Timeouts = grpcclient.Timeouts

                                                                                                        
                                                                                         
func defaultChannelBufferSizes() ChannelBufferSizes {
	return ChannelBufferSizes{
		ResultJSONChan: 10000,                                                                           
		ErrChan:        100,                                        
		RecvChan:       2000,                                                                  
	}
}

                                                  
func defaultTimeouts() Timeouts {
	return Timeouts{
		KeepaliveTime:    300 * time.Second,                                                                        
		KeepaliveTimeout: 20 * time.Second,
		CloseTimeout:     5 * time.Second,
	}
}

                                                                      
  
                                                                               
                                                                      
  
                       
                      
                           
                                   
func NewClient(config ClientConfig) (*Client, error) {
	if config.Endpoint == "" {
		return nil, errors.New("endpoint is required")
	}
	if config.TokenizerPath == "" {
		return nil, errors.New("tokenizer path is required")
	}

	bufferSizes := defaultChannelBufferSizes()
	if config.ChannelBufferSizes != nil {
		if config.ChannelBufferSizes.ResultJSONChan > 0 {
			bufferSizes.ResultJSONChan = config.ChannelBufferSizes.ResultJSONChan
		}
		if config.ChannelBufferSizes.ErrChan > 0 {
			bufferSizes.ErrChan = config.ChannelBufferSizes.ErrChan
		}
		if config.ChannelBufferSizes.RecvChan > 0 {
			bufferSizes.RecvChan = config.ChannelBufferSizes.RecvChan
		}
	}

	timeouts := defaultTimeouts()
	if config.Timeouts != nil {
		if config.Timeouts.KeepaliveTime > 0 {
			timeouts.KeepaliveTime = config.Timeouts.KeepaliveTime
		}
		if config.Timeouts.KeepaliveTimeout > 0 {
			timeouts.KeepaliveTimeout = config.Timeouts.KeepaliveTimeout
		}
		if config.Timeouts.CloseTimeout > 0 {
			timeouts.CloseTimeout = config.Timeouts.CloseTimeout
		}
	}

	grpcClient, err := grpcclient.NewGrpcClient(config.Endpoint, config.TokenizerPath, bufferSizes, timeouts)
	if err != nil {
		return nil, fmt.Errorf("failed to create gRPC client: %w", err)
	}

	return &Client{
		endpoint:      config.Endpoint,
		tokenizerPath: config.TokenizerPath,
		grpcClient:    grpcClient,
	}, nil
}

                                                      
  
                                                                           
                                                         
func (c *Client) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.grpcClient != nil {
		if err := c.grpcClient.Close(); err != nil {
			return err
		}
		c.grpcClient = nil
	}
	return nil
}

                                                                  
                                                      
type ChatCompletionRequest struct {
	                                                                    
	Model string `json:"model"`
	                                                       
	Messages            []ChatMessage   `json:"messages"`
	Temperature         *float32        `json:"temperature,omitempty"`
	TopP                *float32        `json:"top_p,omitempty"`
	TopK                *int            `json:"top_k,omitempty"`
	MaxCompletionTokens *int            `json:"max_completion_tokens,omitempty"`
	Stream              bool            `json:"stream"`
	Tools               []Tool          `json:"tools,omitempty"`
	ToolChoice          interface{}     `json:"tool_choice,omitempty"`
	Stop                interface{}     `json:"stop,omitempty"`
	StopTokenIDs        []int           `json:"stop_token_ids,omitempty"`
	SkipSpecialTokens   bool            `json:"skip_special_tokens,omitempty"`
	FrequencyPenalty    *float32        `json:"frequency_penalty,omitempty"`
	PresencePenalty     *float32        `json:"presence_penalty,omitempty"`
	ResponseFormat      *ResponseFormat `json:"response_format,omitempty"`
	Seed                *int            `json:"seed,omitempty"`
	Logprobs            bool            `json:"logprobs,omitempty"`
	TopLogprobs         *int            `json:"top_logprobs,omitempty"`
	User                string          `json:"user,omitempty"`
}

                                                                 
type ChatMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
	Name    string      `json:"name,omitempty"`
}

                                                     
type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

                                            
type Function struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters"`
}

                                                
type ResponseFormat struct {
	Type string `json:"type"`
}

                                                                             
type ChatCompletionResponse struct {
	ID                string   `json:"id"`
	Object            string   `json:"object"`
	Created           int64    `json:"created"`
	Model             string   `json:"model"`
	SystemFingerprint string   `json:"system_fingerprint,omitempty"`
	Choices           []Choice `json:"choices"`
	Usage             Usage    `json:"usage"`
}

                                                        
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

                                               
type Message struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

                                                  
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

                                          
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

                                           
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

                                                                               
type ChatCompletionStreamResponse struct {
	ID                string         `json:"id"`
	Object            string         `json:"object"`
	Created           int64          `json:"created"`
	Model             string         `json:"model"`
	SystemFingerprint string         `json:"system_fingerprint,omitempty"`
	Choices           []StreamChoice `json:"choices"`
	Usage             *Usage         `json:"usage,omitempty"`
}

                                                           
type StreamChoice struct {
	Index        int          `json:"index"`
	Delta        MessageDelta `json:"delta"`
	FinishReason string       `json:"finish_reason,omitempty"`
}

                                                      
type MessageDelta struct {
	Role      string     `json:"role,omitempty"`
	Content   string     `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

                                                                                     
  
                   
                                                                      
                                                                                            
                                                                       
  
                        
  
                                                                           
                 
                                                     
  
                                                                   
                                                    
func (c *Client) CreateChatCompletion(ctx context.Context, req ChatCompletionRequest) (*ChatCompletionResponse, error) {
	                                                                            
	req.Stream = true                                                             

	if len(req.Tools) == 0 {
		req.Tools = nil
	}

	stream, err := c.CreateChatCompletionStream(ctx, req)
	if err != nil {
		return nil, err
	}
	defer stream.Close()

	var fullContent strings.Builder
	var fullToolCalls []ToolCall
	var finishReason string
	var usage Usage
	var responseID string
	var created int64
	var model string
	var systemFingerprint string

	for {
		chunkJSON, err := stream.RecvJSON()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		var chunk ChatCompletionStreamResponse
		if err := json.Unmarshal([]byte(chunkJSON), &chunk); err != nil {
			return nil, fmt.Errorf("failed to parse chunk: %w", err)
		}

		if chunk.ID != "" {
			responseID = chunk.ID
		}
		if chunk.Created > 0 {
			created = chunk.Created
		}
		if chunk.Model != "" {
			model = chunk.Model
		}
		if chunk.SystemFingerprint != "" {
			systemFingerprint = chunk.SystemFingerprint
		}

		for _, choice := range chunk.Choices {
			if choice.Delta.Content != "" {
				fullContent.WriteString(choice.Delta.Content)
			}
			if len(choice.Delta.ToolCalls) > 0 {
				fullToolCalls = append(fullToolCalls, choice.Delta.ToolCalls...)
			}
			if choice.FinishReason != "" {
				finishReason = choice.FinishReason
			}
		}

		if chunk.Usage != nil {
			usage = *chunk.Usage
		}
	}

	message := Message{
		Role:    "assistant",
		Content: fullContent.String(),
	}
	if len(fullToolCalls) > 0 {
		message.ToolCalls = fullToolCalls
	}

	if finishReason == "" {
		finishReason = "stop"
	}

	return &ChatCompletionResponse{
		ID:                responseID,
		Object:            "chat.completion",
		Created:           created,
		Model:             model,
		SystemFingerprint: systemFingerprint,
		Choices: []Choice{
			{
				Index:        0,
				Message:      message,
				FinishReason: finishReason,
			},
		},
		Usage: usage,
	}, nil
}

                                                              
type ChatCompletionStream struct {
	grpcStream *grpcclient.GrpcChatCompletionStream
	ctx        context.Context
	cancel     context.CancelFunc
}

func (s *ChatCompletionStream) RecvJSON() (string, error) {
	return s.grpcStream.RecvJSON()
}

                                                              
func (s *ChatCompletionStream) Close() error {
	if s.cancel != nil {
		s.cancel()
	}
	if s.grpcStream != nil {
		return s.grpcStream.Close()
	}
	return nil
}

                                                                                                    
  
                   
                                                                          
                                                                                         
                                                                                           
                                                    
  
                        
  
                                                                           
                 
                                                             
                                                 
  
                             
  
                                                          
                                                             
              
                                
                                          
      
func (c *Client) CreateChatCompletionStream(ctx context.Context, req ChatCompletionRequest) (*ChatCompletionStream, error) {
	reqJSON, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	var reqMap map[string]interface{}
	if err := json.Unmarshal(reqJSON, &reqMap); err != nil {
		return nil, fmt.Errorf("failed to unmarshal request to map: %w", err)
	}

	if _, exists := reqMap["tools"]; !exists {
		reqMap["tools"] = []interface{}{}
	}

	reqJSON, err = json.Marshal(reqMap)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request map to JSON: %w", err)
	}

	if c.grpcClient == nil {
		return nil, errors.New("gRPC client is closed")
	}

	grpcStream, err := c.grpcClient.CreateChatCompletionStream(ctx, string(reqJSON))
	if err != nil {
		return nil, fmt.Errorf("failed to create gRPC stream: %w", err)
	}

	streamCtx, cancel := context.WithCancel(ctx)
	return &ChatCompletionStream{
		grpcStream: grpcStream,
		ctx:        streamCtx,
		cancel:     cancel,
	}, nil
}

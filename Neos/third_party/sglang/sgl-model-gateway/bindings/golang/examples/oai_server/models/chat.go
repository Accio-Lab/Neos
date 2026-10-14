package models

                                                                      
type ChatRequest struct {
	Model                string                   `json:"model" binding:"required"`
	Messages             []map[string]string      `json:"messages" binding:"required"`
	Stream               bool                     `json:"stream,omitempty"`
	Temperature          *float64                 `json:"temperature,omitempty"`
	TopP                 *float64                 `json:"top_p,omitempty"`
	MaxTokens            *int                     `json:"max_tokens,omitempty"`                                          
	MaxCompletionTokens *int                     `json:"max_completion_tokens,omitempty"`                                                       
	Tools                []map[string]interface{} `json:"tools,omitempty"`
	ToolChoice           interface{}              `json:"tool_choice,omitempty"`
}

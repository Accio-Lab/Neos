package handlers

import (
	"encoding/json"

	"github.com/valyala/fasthttp"
	"go.uber.org/zap"
)

                                            
type ModelsHandler struct {
	logger        *zap.Logger
	tokenizerPath string
}

                                                
func NewModelsHandler(logger *zap.Logger, tokenizerPath string) *ModelsHandler {
	return &ModelsHandler{
		logger:        logger,
		tokenizerPath: tokenizerPath,
	}
}

                              
func (h *ModelsHandler) List(ctx *fasthttp.RequestCtx) {
	                                                  
	ctx.SetStatusCode(200)
	ctx.SetContentType("application/json")

	response := map[string]interface{}{
		"object": "list",
		"data": []map[string]interface{}{
			{
				"id":       "default",
				"object":   "model",
				"created": 1677610602,
				"owned_by": "sglang",
			},
		},
	}

	jsonData, _ := json.Marshal(response)
	ctx.Write(jsonData)
}

                                           
                                                                   
func (h *ModelsHandler) GetModelInfo(ctx *fasthttp.RequestCtx) {
	ctx.SetStatusCode(200)
	ctx.SetContentType("application/json")

	                                                                        
	response := map[string]interface{}{
		"model_path": h.tokenizerPath,                                    
		"tokenizer_path": h.tokenizerPath,
		"is_generation": true,
		"preferred_sampling_params": "",
		"weight_version": "",
		"has_image_understanding": false,
		"has_audio_understanding": false,
		"model_type": "",
		"architectures": nil,
	}

	jsonData, _ := json.Marshal(response)
	ctx.Write(jsonData)
}

package utils

import (
	"encoding/json"

	"github.com/valyala/fasthttp"
)

                                                        
func RespondError(ctx *fasthttp.RequestCtx, statusCode int, message, errorType string) {
	ctx.SetStatusCode(statusCode)
	ctx.SetContentType("application/json")

	response := map[string]interface{}{
		"error": map[string]interface{}{
			"message": message,
			"type":    errorType,
			"code":    statusCode,
		},
	}

	jsonData, _ := json.Marshal(response)
	ctx.Write(jsonData)
}

                                                                                       
func BuildResponseBase(id string, created int64, model string) map[string]interface{} {
	return map[string]interface{}{
		"id":      id,
		"object":  "chat.completion",
		"created": created,
		"model":   model,
	}
}

package handlers

import (
	"encoding/json"

	"github.com/valyala/fasthttp"
	"go.uber.org/zap"
)

                                              
type HealthHandler struct {
	logger *zap.Logger
}

                                                
func NewHealthHandler(logger *zap.Logger) *HealthHandler {
	return &HealthHandler{
		logger: logger,
	}
}

                            
func (h *HealthHandler) Check(ctx *fasthttp.RequestCtx) {
	ctx.SetStatusCode(200)
	ctx.SetContentType("application/json")

	response := map[string]string{
		"status": "ok",
	}

	jsonData, _ := json.Marshal(response)
	ctx.Write(jsonData)
}

package config

import (
	"os"
)

                                             
type Config struct {
	Endpoint      string
	TokenizerPath string
	Port          string
	LogDir        string
	LogLevel      string
}

                                                                    
func Load() *Config {
	                                                     
	tokenizerPath := os.Getenv("SGL_TOKENIZER_PATH")
	if tokenizerPath == "" {
		tokenizerPath = "../tokenizer"
	}

	                                               
	endpoint := os.Getenv("SGL_GRPC_ENDPOINT")
	if endpoint == "" {
		endpoint = "grpc://localhost:20000"
	}

	                                           
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	                                                    
	logDir := os.Getenv("LOG_DIR")
	if logDir == "" {
		logDir = "./logs"
	}

	                                                
	logLevel := os.Getenv("LOG_LEVEL")
	if logLevel == "" {
		logLevel = "info"
	}

	return &Config{
		Endpoint:      endpoint,
		TokenizerPath: tokenizerPath,
		Port:          port,
		LogDir:        logDir,
		LogLevel:      logLevel,
	}
}

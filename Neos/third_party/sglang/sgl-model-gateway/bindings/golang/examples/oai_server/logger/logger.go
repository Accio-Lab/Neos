package logger

import (
	"os"
	"path/filepath"
	"time"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"gopkg.in/natefinch/lumberjack.v2"
)

                                                           
func Init(logDir, logLevel string) (*zap.Logger, error) {
	                              
	if err := os.MkdirAll(logDir, 0755); err != nil {
		return nil, err
	}

	                  
	var level zapcore.Level
	if err := level.UnmarshalText([]byte(logLevel)); err != nil {
		level = zapcore.InfoLevel
	}

	                                 
	logFile := filepath.Join(logDir, "oai_server-"+time.Now().Format("2006-01-02")+".log")

	                            
	fileWriter := zapcore.AddSync(&lumberjack.Logger{
		Filename:   logFile,
		MaxSize:    100,             
		MaxBackups: 10,
		MaxAge:     30,        
		Compress:   true,
	})

	                 
	consoleWriter := zapcore.AddSync(os.Stdout)

	                 
	encoderConfig := zap.NewProductionEncoderConfig()
	encoderConfig.TimeKey = "timestamp"
	encoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
	encoderConfig.EncodeLevel = zapcore.CapitalLevelEncoder

	               
	fileCore := zapcore.NewCore(
		zapcore.NewJSONEncoder(encoderConfig),
		fileWriter,
		level,
	)

	consoleCore := zapcore.NewCore(
		zapcore.NewConsoleEncoder(encoderConfig),
		consoleWriter,
		level,
	)

	                
	core := zapcore.NewTee(fileCore, consoleCore)

	                
	logger := zap.New(core, zap.AddCaller(), zap.AddStacktrace(zapcore.ErrorLevel))

	return logger, nil
}

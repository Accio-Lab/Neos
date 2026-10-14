package service

import (
	sglang "github.com/sglang/sglang-go-grpc-sdk"
)

                                    
type SGLangService struct {
	client *sglang.Client
}

func NewSGLangService(endpoint, tokenizerPath string) (*SGLangService, error) {
	client, err := sglang.NewClient(sglang.ClientConfig{
		Endpoint:      endpoint,
		TokenizerPath: tokenizerPath,
	})
	if err != nil {
		return nil, err
	}

	return &SGLangService{
		client: client,
	}, nil
}

                                              
func (s *SGLangService) Client() *sglang.Client {
	return s.client
}

                                 
func (s *SGLangService) Close() error {
	if s.client != nil {
		return s.client.Close()
	}
	return nil
}

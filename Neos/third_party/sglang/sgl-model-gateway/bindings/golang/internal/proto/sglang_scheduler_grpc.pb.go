                                                     
            
                              
                                
                                 

package proto

import (
	context "context"
	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
)

                                                                      
                                                                    
                                     
const _ = grpc.SupportPackageIsVersion9

const (
	SglangScheduler_Generate_FullMethodName      = "/sglang.grpc.scheduler.SglangScheduler/Generate"
	SglangScheduler_Embed_FullMethodName         = "/sglang.grpc.scheduler.SglangScheduler/Embed"
	SglangScheduler_HealthCheck_FullMethodName   = "/sglang.grpc.scheduler.SglangScheduler/HealthCheck"
	SglangScheduler_Abort_FullMethodName         = "/sglang.grpc.scheduler.SglangScheduler/Abort"
	SglangScheduler_GetModelInfo_FullMethodName  = "/sglang.grpc.scheduler.SglangScheduler/GetModelInfo"
	SglangScheduler_GetServerInfo_FullMethodName = "/sglang.grpc.scheduler.SglangScheduler/GetServerInfo"
)

                                                                       
  
                                                                                                                                                           
  
                                                        
                                                             
type SglangSchedulerClient interface {
	                                                   
	Generate(ctx context.Context, in *GenerateRequest, opts ...grpc.CallOption) (grpc.ServerStreamingClient[GenerateResponse], error)
	                              
	Embed(ctx context.Context, in *EmbedRequest, opts ...grpc.CallOption) (*EmbedResponse, error)
	                           
	HealthCheck(ctx context.Context, in *HealthCheckRequest, opts ...grpc.CallOption) (*HealthCheckResponse, error)
	                          
	Abort(ctx context.Context, in *AbortRequest, opts ...grpc.CallOption) (*AbortResponse, error)
	                        
	GetModelInfo(ctx context.Context, in *GetModelInfoRequest, opts ...grpc.CallOption) (*GetModelInfoResponse, error)
	                         
	GetServerInfo(ctx context.Context, in *GetServerInfoRequest, opts ...grpc.CallOption) (*GetServerInfoResponse, error)
}

type sglangSchedulerClient struct {
	cc grpc.ClientConnInterface
}

func NewSglangSchedulerClient(cc grpc.ClientConnInterface) SglangSchedulerClient {
	return &sglangSchedulerClient{cc}
}

func (c *sglangSchedulerClient) Generate(ctx context.Context, in *GenerateRequest, opts ...grpc.CallOption) (grpc.ServerStreamingClient[GenerateResponse], error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	stream, err := c.cc.NewStream(ctx, &SglangScheduler_ServiceDesc.Streams[0], SglangScheduler_Generate_FullMethodName, cOpts...)
	if err != nil {
		return nil, err
	}
	x := &grpc.GenericClientStream[GenerateRequest, GenerateResponse]{ClientStream: stream}
	if err := x.ClientStream.SendMsg(in); err != nil {
		return nil, err
	}
	if err := x.ClientStream.CloseSend(); err != nil {
		return nil, err
	}
	return x, nil
}

                                                                                                                                        
type SglangScheduler_GenerateClient = grpc.ServerStreamingClient[GenerateResponse]

func (c *sglangSchedulerClient) Embed(ctx context.Context, in *EmbedRequest, opts ...grpc.CallOption) (*EmbedResponse, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(EmbedResponse)
	err := c.cc.Invoke(ctx, SglangScheduler_Embed_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *sglangSchedulerClient) HealthCheck(ctx context.Context, in *HealthCheckRequest, opts ...grpc.CallOption) (*HealthCheckResponse, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(HealthCheckResponse)
	err := c.cc.Invoke(ctx, SglangScheduler_HealthCheck_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *sglangSchedulerClient) Abort(ctx context.Context, in *AbortRequest, opts ...grpc.CallOption) (*AbortResponse, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(AbortResponse)
	err := c.cc.Invoke(ctx, SglangScheduler_Abort_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *sglangSchedulerClient) GetModelInfo(ctx context.Context, in *GetModelInfoRequest, opts ...grpc.CallOption) (*GetModelInfoResponse, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(GetModelInfoResponse)
	err := c.cc.Invoke(ctx, SglangScheduler_GetModelInfo_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *sglangSchedulerClient) GetServerInfo(ctx context.Context, in *GetServerInfoRequest, opts ...grpc.CallOption) (*GetServerInfoResponse, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(GetServerInfoResponse)
	err := c.cc.Invoke(ctx, SglangScheduler_GetServerInfo_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

                                                                       
                                                                    
                             
  
                                                        
                                                             
type SglangSchedulerServer interface {
	                                                   
	Generate(*GenerateRequest, grpc.ServerStreamingServer[GenerateResponse]) error
	                              
	Embed(context.Context, *EmbedRequest) (*EmbedResponse, error)
	                           
	HealthCheck(context.Context, *HealthCheckRequest) (*HealthCheckResponse, error)
	                          
	Abort(context.Context, *AbortRequest) (*AbortResponse, error)
	                        
	GetModelInfo(context.Context, *GetModelInfoRequest) (*GetModelInfoResponse, error)
	                         
	GetServerInfo(context.Context, *GetServerInfoRequest) (*GetServerInfoResponse, error)
	mustEmbedUnimplementedSglangSchedulerServer()
}

                                                              
                                      
  
                                                                           
                                               
type UnimplementedSglangSchedulerServer struct{}

func (UnimplementedSglangSchedulerServer) Generate(*GenerateRequest, grpc.ServerStreamingServer[GenerateResponse]) error {
	return status.Errorf(codes.Unimplemented, "method Generate not implemented")
}
func (UnimplementedSglangSchedulerServer) Embed(context.Context, *EmbedRequest) (*EmbedResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Embed not implemented")
}
func (UnimplementedSglangSchedulerServer) HealthCheck(context.Context, *HealthCheckRequest) (*HealthCheckResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method HealthCheck not implemented")
}
func (UnimplementedSglangSchedulerServer) Abort(context.Context, *AbortRequest) (*AbortResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Abort not implemented")
}
func (UnimplementedSglangSchedulerServer) GetModelInfo(context.Context, *GetModelInfoRequest) (*GetModelInfoResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method GetModelInfo not implemented")
}
func (UnimplementedSglangSchedulerServer) GetServerInfo(context.Context, *GetServerInfoRequest) (*GetServerInfoResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method GetServerInfo not implemented")
}
func (UnimplementedSglangSchedulerServer) mustEmbedUnimplementedSglangSchedulerServer() {}
func (UnimplementedSglangSchedulerServer) testEmbeddedByValue()                         {}

                                                                                                    
                                                                                           
                                
type UnsafeSglangSchedulerServer interface {
	mustEmbedUnimplementedSglangSchedulerServer()
}

func RegisterSglangSchedulerServer(s grpc.ServiceRegistrar, srv SglangSchedulerServer) {
	                                                                                    
	                                                                
	                                                                          
	                                                                 
	if t, ok := srv.(interface{ testEmbeddedByValue() }); ok {
		t.testEmbeddedByValue()
	}
	s.RegisterService(&SglangScheduler_ServiceDesc, srv)
}

func _SglangScheduler_Generate_Handler(srv interface{}, stream grpc.ServerStream) error {
	m := new(GenerateRequest)
	if err := stream.RecvMsg(m); err != nil {
		return err
	}
	return srv.(SglangSchedulerServer).Generate(m, &grpc.GenericServerStream[GenerateRequest, GenerateResponse]{ServerStream: stream})
}

                                                                                                                                        
type SglangScheduler_GenerateServer = grpc.ServerStreamingServer[GenerateResponse]

func _SglangScheduler_Embed_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(EmbedRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(SglangSchedulerServer).Embed(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: SglangScheduler_Embed_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(SglangSchedulerServer).Embed(ctx, req.(*EmbedRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _SglangScheduler_HealthCheck_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(HealthCheckRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(SglangSchedulerServer).HealthCheck(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: SglangScheduler_HealthCheck_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(SglangSchedulerServer).HealthCheck(ctx, req.(*HealthCheckRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _SglangScheduler_Abort_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(AbortRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(SglangSchedulerServer).Abort(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: SglangScheduler_Abort_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(SglangSchedulerServer).Abort(ctx, req.(*AbortRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _SglangScheduler_GetModelInfo_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(GetModelInfoRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(SglangSchedulerServer).GetModelInfo(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: SglangScheduler_GetModelInfo_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(SglangSchedulerServer).GetModelInfo(ctx, req.(*GetModelInfoRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _SglangScheduler_GetServerInfo_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(GetServerInfoRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(SglangSchedulerServer).GetServerInfo(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: SglangScheduler_GetServerInfo_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(SglangSchedulerServer).GetServerInfo(ctx, req.(*GetServerInfoRequest))
	}
	return interceptor(ctx, in, info, handler)
}

                                                                                   
                                                               
                                                          
var SglangScheduler_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "sglang.grpc.scheduler.SglangScheduler",
	HandlerType: (*SglangSchedulerServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "Embed",
			Handler:    _SglangScheduler_Embed_Handler,
		},
		{
			MethodName: "HealthCheck",
			Handler:    _SglangScheduler_HealthCheck_Handler,
		},
		{
			MethodName: "Abort",
			Handler:    _SglangScheduler_Abort_Handler,
		},
		{
			MethodName: "GetModelInfo",
			Handler:    _SglangScheduler_GetModelInfo_Handler,
		},
		{
			MethodName: "GetServerInfo",
			Handler:    _SglangScheduler_GetServerInfo_Handler,
		},
	},
	Streams: []grpc.StreamDesc{
		{
			StreamName:    "Generate",
			Handler:       _SglangScheduler_Generate_Handler,
			ServerStreams: true,
		},
	},
	Metadata: "sglang_scheduler.proto",
}

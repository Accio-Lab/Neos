                                                                                             
 
                                                                      
                    
 
          
                                                                                                    
                                      
                                                                                             
                            
 
macro(clear_cuda_arches CUDA_ARCH_FLAGS)
                                                          
    string(REGEX MATCHALL "-gencode arch=[^ ]+" CUDA_ARCH_FLAGS "${CMAKE_CUDA_FLAGS}")

                                                                                     
                                                            
    string(REGEX REPLACE "-gencode arch=[^ ]+ *" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
endmacro()

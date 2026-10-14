                                                                     

                                                                                         
                                                                              

#pragma once

                                                         
                                                                     
                                                         
   
          
       
                                      
                                      
       
       
#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      static constexpr bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      static constexpr bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

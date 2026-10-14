#pragma once

#include <climits>
#include <iostream>

inline constexpr uint32_t next_pow_2(uint32_t const num) {
  if (num <= 1) return num;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

template <typename A, typename B>
static inline constexpr auto div_ceil(A a, B b) {
  return (a + b - 1) / b;
}

                                                                               
                          
template <typename T>
inline constexpr T round_to_previous_multiple_of(T a, T b) {
  return a % b == 0 ? a : (a / b) * b;
}

                                                                             
                          
template <typename T>
inline constexpr T round_to_next_multiple_of(T a, T b) {
  return a % b == 0 ? a : ((a / b) + 1) * b;
}

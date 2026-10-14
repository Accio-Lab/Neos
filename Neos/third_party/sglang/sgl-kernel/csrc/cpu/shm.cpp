#include "shm.h"
#if defined(__x86_64__)
#include "x86_64/shm.h"
#elif defined(__aarch64__)
#include "aarch64/shm.h"
#else
#error "unsupported architecture"
#endif

#include <ATen/ATen.h>
#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

                         
enum coll_state {
  coll_begin = 0,
  coll_allreduce_naive__copy_in_done,
  coll_allreduce_naive__reduce_done,
                                                                      
                          
  coll_alt1_allreduce_naive__copy_in_done,
  coll_alt2_allreduce_naive__copy_in_done,
  coll_alt1_allreduce_naive__reduce_done,
  coll_allgather_naive__copy_in_done,
  coll_alt1_allgather_naive__copy_in_done,
  coll_alt2_allgather_naive__copy_in_done,
};

                      
struct SharedData {
  const char* name;
  int descriptor;
  void* bytes;
  size_t nbytes;
};

void shared_open(SharedData* data, const char* name, size_t nbytes) {
  int d = shm_open(name, O_RDWR, S_IRUSR | S_IWUSR);
  if (d != -1) {
    void* bytes = mmap(NULL, nbytes, PROT_READ | PROT_WRITE, MAP_SHARED, d, 0);
    data->name = name;
    data->descriptor = d;
    data->bytes = bytes;
    data->nbytes = nbytes;
  } else {
    if (errno != ENOENT) {
                                                                              
                                                           
      printf("shared_open %s failed, errno=%d\n", name, errno);
    }
    data->descriptor = -1;
  }
}

void shared_create(SharedData* data, const char* name, void* bytes, size_t nbytes) {
  int d = shm_open(name, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  if (d != -1) {
    nbytes = write(d, bytes, nbytes);
    if (nbytes > 0) {
      shared_open(data, name, nbytes);
    }
  } else {
    printf("shared_create %s failed\n", name);
  }
}

static int world_size;

                                       
                             
#define NAME_BUF_SIZE 1000
#define MAX_BUF_SIZE 1048576 * 32
#define NAIVE_ALLREDUCE_THRESHOLD 1048576
#define SHM_BUFFER_NAME "deepspeed_allreduce_buffer"
struct allreduce_workspace {
  enum coll_state states[2];                                                  
                                                                                
                                                  
                                                         
                                                       
                                 
  char buffer[2 * NAIVE_ALLREDUCE_THRESHOLD + 2 * MAX_BUF_SIZE];
};

#define BUFFER0_OFFSET(current_buffer) current_buffer* NAIVE_ALLREDUCE_THRESHOLD
#define BUFFER1_OFFSET(current_buffer) 2 * NAIVE_ALLREDUCE_THRESHOLD + current_buffer* MAX_BUF_SIZE

struct allreduce_workspace** workspace;

                                           
char** symmetric_buffer[2];
                                           
char** distributed_buffer[2];

void wait_buffer_state_until_2(int index, enum coll_state state0, enum coll_state state1, int state_group) {
  volatile enum coll_state* state_ptr = &(workspace[index]->states[state_group]);

  while (1) {
    volatile enum coll_state cur_state = *state_ptr;
    if (cur_state == state0 || cur_state == state1) break;
  }
}

void reduce_all_buffers(
    int start_elements,
    int num_elements,
    c10::ScalarType scalar_type,
    int to_buffer_idx,
    char* to_buffer,
    char** buffers) {
  switch (scalar_type) {
    case c10::ScalarType::BFloat16:
      reduce_bf16_buffers(start_elements, num_elements, to_buffer, buffers, world_size);
      break;
    case c10::ScalarType::Half:
      reduce_fp16_buffers(start_elements, num_elements, to_buffer, buffers, world_size);
      break;
    case c10::ScalarType::Float:
      reduce_fp32_buffers(start_elements, num_elements, to_buffer, buffers, world_size);
      break;
    default:
      assert(!"Should not get here");
  }
}

static bool is_initialized = false;
static int world_rank;

void shm_initialize(int size, int rank, const char* addr_string, const char* port_string) {
  if (is_initialized) {
    return;
  }
  is_initialized = true;

  world_size = size;
  world_rank = rank;

  char shm_name_prefix[NAME_BUF_SIZE];
  char shm_name[NAME_BUF_SIZE];
  snprintf(shm_name_prefix, NAME_BUF_SIZE, "%s_%d_%s_%s", SHM_BUFFER_NAME, getuid(), addr_string, port_string);
                                                    
  SharedData allreduce_buffer;
                                            
  struct allreduce_workspace* workspace_buf;
  struct allreduce_workspace* workspace_buf_other;
  workspace_buf = (struct allreduce_workspace*)malloc(sizeof(struct allreduce_workspace));
  snprintf(shm_name, NAME_BUF_SIZE, "%.900s_%d", shm_name_prefix, rank);
  shared_create(&allreduce_buffer, shm_name, workspace_buf, sizeof(struct allreduce_workspace));
  workspace_buf = (struct allreduce_workspace*)allreduce_buffer.bytes;
  workspace_buf->states[0] = coll_alt2_allreduce_naive__copy_in_done;
  workspace_buf->states[1] = coll_begin;

                                      
  workspace = (struct allreduce_workspace**)malloc(size * sizeof(struct allreduce_workspace*));
  symmetric_buffer[0] = (char**)malloc(size * sizeof(char**));
  symmetric_buffer[1] = (char**)malloc(size * sizeof(char**));
  distributed_buffer[0] = (char**)malloc(size * sizeof(char**));
  distributed_buffer[1] = (char**)malloc(size * sizeof(char**));

                         
  for (int i = 0; i < size; i++) {
    if (i != rank) {
      snprintf(shm_name, NAME_BUF_SIZE, "%.900s_%d", shm_name_prefix, i);
                                                 
      do {
        shared_open(&allreduce_buffer, shm_name, sizeof(struct allreduce_workspace));
      } while (allreduce_buffer.descriptor == -1 && errno == ENOENT);
      workspace_buf_other = (struct allreduce_workspace*)allreduce_buffer.bytes;
      workspace[i] = workspace_buf_other;
    } else {
      workspace[i] = workspace_buf;
    }
    symmetric_buffer[0][i] = workspace[i]->buffer + BUFFER0_OFFSET(0);
    symmetric_buffer[1][i] = workspace[i]->buffer + BUFFER0_OFFSET(1);
    distributed_buffer[0][i] = workspace[i]->buffer + BUFFER1_OFFSET(0);
    distributed_buffer[1][i] = workspace[i]->buffer + BUFFER1_OFFSET(1);
  }
}

#define positive_mod(num, mod) ((((num) % (mod)) + (mod)) % (mod))
#define rank_mod(rank) positive_mod(rank, world_size)
size_t slice_size(size_t chunk_el, int slice_idx) {
  size_t slice_size = chunk_el / world_size;
  return slice_idx == world_size - 1 ? slice_size + (chunk_el % world_size) : slice_size;
}

char* slice_data(char* data_ptr, size_t chunk_el, int el_size, int slice_idx) {
  size_t slice_size = chunk_el / world_size;
  size_t el_offset = slice_size * slice_idx;
  return data_ptr + el_offset * el_size;
}

size_t slice_el_start(size_t chunk_el, int slice_idx) {
  size_t slice_size = chunk_el / world_size;
  return slice_size * slice_idx;
}

void symmetric_naive_all_reduce(char* data_ptr, c10::ScalarType scalar_type, size_t chunk_size, size_t chunk_el) {
  const int state_group = 0;
  static int current_buffer = 0;
  static int state_idx = 0;

                                                                       
  enum coll_state copy_current = coll_allreduce_naive__copy_in_done;
  enum coll_state copy_next = coll_alt1_allreduce_naive__copy_in_done;

  switch (state_idx) {
    case 0:
      copy_current = coll_allreduce_naive__copy_in_done;
      copy_next = coll_alt1_allreduce_naive__copy_in_done;
      break;
    case 1:
      copy_current = coll_alt1_allreduce_naive__copy_in_done;
      copy_next = coll_alt2_allreduce_naive__copy_in_done;
      break;
    case 2:
      copy_current = coll_alt2_allreduce_naive__copy_in_done;
      copy_next = coll_allreduce_naive__copy_in_done;
      break;
    default:
      assert(!"Should not get here.");
  }
  state_idx = (state_idx + 1) % 3;

  parallel_memcpy(symmetric_buffer[current_buffer][world_rank], data_ptr, chunk_size);
  std::atomic_thread_fence(std::memory_order_release);
  workspace[world_rank]->states[state_group] = copy_current;

  for (int i = 0; i < world_size; i++) {
                                                
    if (i != world_rank) {
      wait_buffer_state_until_2(i, copy_current, copy_next, state_group);
    }
  }

                                                                       
                              
  reduce_all_buffers(0, chunk_el, scalar_type, world_rank, data_ptr, symmetric_buffer[current_buffer]);

                  
  current_buffer = 1 - current_buffer;
}

                                                                      
void distributed_naive_reduce(char* data_ptr, c10::ScalarType scalar_type, size_t chunk_size, size_t chunk_el) {
  const int state_group = 1;
  static int current_buffer = 0;
  static int state_idx = 0;

                                                                       
  enum coll_state copy_current = coll_allreduce_naive__copy_in_done;
  enum coll_state reduce_current = coll_allreduce_naive__reduce_done;
  enum coll_state copy_next = coll_alt1_allreduce_naive__copy_in_done;

                                                                            
                                                                               
  switch (state_idx) {
    case 0:
      copy_current = coll_allreduce_naive__copy_in_done;
      reduce_current = coll_allreduce_naive__reduce_done;
      copy_next = coll_alt1_allreduce_naive__copy_in_done;
      break;
    case 1:
      copy_current = coll_alt1_allreduce_naive__copy_in_done;
      reduce_current = coll_alt1_allreduce_naive__reduce_done;
      copy_next = coll_allreduce_naive__copy_in_done;
      break;
    default:
      assert(!"Should not get here.");
  }
  state_idx = (state_idx + 1) % 2;

  int data_size = chunk_size / chunk_el;
  parallel_memcpy(distributed_buffer[current_buffer][world_rank], data_ptr, chunk_size);
  std::atomic_thread_fence(std::memory_order_release);
  workspace[world_rank]->states[state_group] = copy_current;

  for (int i = 0; i < world_size; i++) {
                                                     
    if (i != world_rank) wait_buffer_state_until_2(i, copy_current, reduce_current, state_group);
  }

                   
  reduce_all_buffers(
      slice_el_start(chunk_el, world_rank),
      slice_size(chunk_el, world_rank),
      scalar_type,
      world_rank,
      distributed_buffer[current_buffer][world_rank],
      distributed_buffer[current_buffer]);
  std::atomic_thread_fence(std::memory_order_release);
  workspace[world_rank]->states[state_group] = reduce_current;

  for (int i = 0; i < world_size; i++) {
                                                       
    if (i != world_rank) wait_buffer_state_until_2(i, reduce_current, copy_next, state_group);
  }

  for (int i = 0; i < world_size; i++) {
    int rank = (i + world_rank) % world_size;
    parallel_memcpy(
        slice_data(data_ptr, chunk_el, data_size, rank),
        slice_data(distributed_buffer[current_buffer][rank], chunk_el, chunk_size / chunk_el, rank),
        slice_size(chunk_el, rank) * data_size);
  }

  current_buffer = 1 - current_buffer;
}

void all_reduce_outer_loop(torch::Tensor& data, size_t numel, int data_size) {
  for (int offset = 0; offset < data_size; offset += MAX_BUF_SIZE) {
    auto data_ptr = ((char*)(data.data_ptr()) + offset);
    size_t chunk_size = data_size - offset > MAX_BUF_SIZE ? MAX_BUF_SIZE : data_size - offset;
    size_t chunk_el = chunk_size / (data_size / numel);
    if (chunk_size < NAIVE_ALLREDUCE_THRESHOLD) {
      symmetric_naive_all_reduce(data_ptr, data.scalar_type(), chunk_size, chunk_el);
    } else {
      distributed_naive_reduce(data_ptr, data.scalar_type(), chunk_size, chunk_el);
    }
  }
}

void naive_all_gather(char* result_ptr, char* data_ptr, size_t res_stride, size_t chunk_size, size_t chunk_el) {
  const int state_group = 1;
  static int current_buffer = 0;
  static int state_idx = 0;

                                                                       
  enum coll_state copy_current = coll_allgather_naive__copy_in_done;
  enum coll_state copy_next = coll_alt1_allgather_naive__copy_in_done;

  switch (state_idx) {
    case 0:
      copy_current = coll_allgather_naive__copy_in_done;
      copy_next = coll_alt1_allgather_naive__copy_in_done;
      break;
    case 1:
      copy_current = coll_alt1_allgather_naive__copy_in_done;
      copy_next = coll_alt2_allgather_naive__copy_in_done;
      break;
    case 2:
      copy_current = coll_alt2_allgather_naive__copy_in_done;
      copy_next = coll_allgather_naive__copy_in_done;
      break;
    default:
      assert(!"Should not get here.");
  }
  state_idx = (state_idx + 1) % 3;

  parallel_memcpy(distributed_buffer[current_buffer][world_rank], data_ptr, chunk_size);
  std::atomic_thread_fence(std::memory_order_release);
  workspace[world_rank]->states[state_group] = copy_current;

  for (int i = 0; i < world_size; i++) {
                                                     
    if (i != world_rank) wait_buffer_state_until_2(i, copy_current, copy_next, state_group);
  }
  for (int i = 0; i < world_size; i++) {
    parallel_memcpy(result_ptr + i * res_stride, distributed_buffer[current_buffer][i], chunk_size);
  }
  current_buffer = 1 - current_buffer;
}

torch::Tensor& all_gather(torch::Tensor& result, torch::Tensor& data, int dim, size_t numel, int data_size) {
  size_t dim_el = data.stride(dim) * data.size(dim);
  int dtype_size = data_size / numel;
  size_t dim_size = dim_el * dtype_size;
  int dim_count = data_size / dim_size;
  auto data_ptr = (char*)(data.data_ptr());
  auto result_ptr = (char*)(result.data_ptr());
  for (int i = 0; i < dim_count; i++) {
    for (size_t offset = 0; offset < dim_size; offset += MAX_BUF_SIZE) {
      size_t chunk_size = dim_size - offset > MAX_BUF_SIZE ? MAX_BUF_SIZE : dim_size - offset;
      size_t chunk_el = chunk_size / dtype_size;
      naive_all_gather(
          result_ptr + i * dim_size * world_size + offset,
          data_ptr + i * dim_size + offset,
          dim_size,
          chunk_size,
          chunk_el);
    }
  }
  return result;
}

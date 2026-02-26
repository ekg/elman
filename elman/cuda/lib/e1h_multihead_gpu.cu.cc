/**
 * E1H Multi-Head Elman CUDA Kernel
 *
 * Multi-head version of the classic E1 gated Elman recurrence.
 * Each head has an independent hidden state vector h[N] and weight matrix W_h[N,N].
 *
 * Per head h:
 *   raw = pre_x[t,b,h] + W_h @ h_prev + b
 *   h_t = tanh(raw)
 *   out_t = h_t * silu(z[t,b,h])
 *
 * Where:
 *   pre_x: [T, B, H, N]  -- pre-computed input projections (W_x @ x done externally)
 *   z:     [T, B, H, N]  -- gate values (computed externally)
 *   W_h:   [H, N, N]     -- per-head recurrence weight (constant across batch/time)
 *   b:     [H, N]        -- per-head bias (constant across batch/time)
 *   h:     [B, H, N]     -- hidden state vector
 *
 * silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * Backward accumulates:
 *   d_W_h: [B, H, N, N] in float32 (sum across batch in Python)
 *   d_b:   [B, H, N]    in float32 (sum across batch in Python)
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include "hasty/elman_ladder.h"

#define E1H_CHECKPOINT_INTERVAL 16

namespace elman {

// ============================================================================
// E1H Forward Kernel
// Each block handles one (batch, head) pair
// Thread count = N (one thread per state element)
// ============================================================================

template<int N>
__global__ void E1HForwardKernel_BF16(
    int T,
    int B,
    int H,
    const __nv_bfloat16* __restrict__ pre_x_all,   // [T, B, H, N]
    const __nv_bfloat16* __restrict__ z_all,        // [T, B, H, N]
    __nv_bfloat16* __restrict__ h_state,            // [B, H, N] initial state, updated in-place to final
    const __nv_bfloat16* __restrict__ W_h,          // [H, N, N]
    const __nv_bfloat16* __restrict__ b_h,          // [H, N]
    __nv_bfloat16* __restrict__ output,             // [T, B, H, N]
    __nv_bfloat16* __restrict__ h_checkpoints,      // [num_checkpoints, B, H, N]
    int checkpoint_interval
) {
    int block_idx = blockIdx.x;
    int b = block_idx / H;
    int h = block_idx % H;
    if (b >= B) return;

    int tid = threadIdx.x;

    extern __shared__ float shared_mem[];
    // Shared memory layout:
    // W_h_shared: [N * N]  -- loaded once, constant
    // b_shared:   [N]      -- loaded once, constant
    // h_shared:   [N]      -- current hidden state
    // pre_x_shared: [N]    -- current timestep input
    // z_shared:   [N]      -- current timestep gate
    float* W_h_shared = shared_mem;
    float* b_shared = W_h_shared + N * N;
    float* h_shared = b_shared + N;
    float* pre_x_shared = h_shared + N;
    float* z_shared = pre_x_shared + N;

    // Load W_h for this head into shared memory (constant for all timesteps)
    int wh_offset = h * N * N;
    for (int i = tid; i < N * N; i += blockDim.x) {
        W_h_shared[i] = __bfloat162float(W_h[wh_offset + i]);
    }

    // Load bias for this head
    if (tid < N) {
        b_shared[tid] = __bfloat162float(b_h[h * N + tid]);
    }

    // Load initial hidden state
    int state_offset = (b * H + h) * N;
    if (tid < N) {
        h_shared[tid] = __bfloat162float(h_state[state_offset + tid]);
    }
    __syncthreads();

    // Save initial checkpoint (index 0)
    int cp0_offset = (b * H + h) * N;
    if (tid < N) {
        h_checkpoints[cp0_offset + tid] = __float2bfloat16(h_shared[tid]);
    }
    __syncthreads();

    for (int t = 0; t < T; t++) {
        // Compute offset for this timestep
        int io_offset = ((t * B + b) * H + h) * N;

        // Load pre_x and z for this timestep
        if (tid < N) {
            pre_x_shared[tid] = __bfloat162float(pre_x_all[io_offset + tid]);
            z_shared[tid] = __bfloat162float(z_all[io_offset + tid]);
        }
        __syncthreads();

        // Compute h_prev @ W_h: result[j] = sum_i h_prev[i] * W_h[i, j]
        // Thread tid computes element tid of the result
        if (tid < N) {
            float wh_dot = 0.0f;
            #pragma unroll 8
            for (int i = 0; i < N; i++) {
                wh_dot += h_shared[i] * W_h_shared[i * N + tid];
            }

            // h_new = tanh(pre_x + h_prev @ W_h + b)
            float raw = pre_x_shared[tid] + wh_dot + b_shared[tid];
            float h_new = tanhf(raw);

            // output = h_new * silu(z)
            float z_val = z_shared[tid];
            float sig_z = 1.0f / (1.0f + expf(-z_val));
            float silu_z = z_val * sig_z;

            output[io_offset + tid] = __float2bfloat16(h_new * silu_z);
            h_shared[tid] = h_new;
        }
        __syncthreads();

        // Save checkpoint periodically
        if ((t + 1) % checkpoint_interval == 0) {
            int cp_idx = (t + 1) / checkpoint_interval;
            int cp_offset = (cp_idx * B * H + b * H + h) * N;
            if (tid < N) {
                h_checkpoints[cp_offset + tid] = __float2bfloat16(h_shared[tid]);
            }
        }
    }

    // Write final hidden state back
    if (tid < N) {
        h_state[state_offset + tid] = __float2bfloat16(h_shared[tid]);
    }
}

// ============================================================================
// E1H Backward Kernel (Segment-Level Caching)
//
// Uses checkpoint + segment replay approach (same as E88):
//   Phase 1: Replay forward through segment from checkpoint, cache h_{t-1}
//   Phase 2: Backward through segment using cached h_{t-1}
//
// Gradients computed:
//   d_pre_x[t] = d_raw[t]
//   d_z[t]     = d_output[t] * h_t * dsilu(z_t)
//   d_W_h     += outer(d_raw[t], h_{t-1})   -- accumulated across time
//   d_b       += d_raw[t]                    -- accumulated across time
//   d_h_prev   = W_h^T @ d_raw[t]           -- propagated backward
// ============================================================================

template<int N>
__global__ void E1HBackwardKernel_BF16(
    int T,
    int B,
    int H,
    const __nv_bfloat16* __restrict__ pre_x_all,   // [T, B, H, N]
    const __nv_bfloat16* __restrict__ z_all,        // [T, B, H, N]
    const __nv_bfloat16* __restrict__ W_h,          // [H, N, N]
    const __nv_bfloat16* __restrict__ b_h,          // [H, N]
    const __nv_bfloat16* __restrict__ h_checkpoints,// [num_checkpoints, B, H, N]
    const __nv_bfloat16* __restrict__ d_output,     // [T, B, H, N]
    __nv_bfloat16* __restrict__ d_pre_x_all,        // [T, B, H, N]
    __nv_bfloat16* __restrict__ d_z_all,            // [T, B, H, N]
    float* __restrict__ d_W_h,                      // [B, H, N, N] float32
    float* __restrict__ d_b_h,                      // [B, H, N] float32
    __nv_bfloat16* __restrict__ segment_cache,      // [B*H, checkpoint_interval, N] for h_{t-1}
    int checkpoint_interval
) {
    int block_idx = blockIdx.x;
    int b = block_idx / H;
    int h = block_idx % H;
    if (b >= B) return;

    int tid = threadIdx.x;

    extern __shared__ float shared_mem[];
    // Shared memory layout:
    // W_h_shared:    [N * N]  -- constant
    // b_shared:      [N]      -- constant (needed for forward replay)
    // h_prev_shared: [N]      -- h_{t-1} during backward
    // h_t_shared:    [N]      -- recomputed h_t
    // d_h_shared:    [N]      -- gradient propagating backward through time
    // d_W_h_acc:     [N * N]  -- accumulated gradient for W_h (float32)
    // d_b_acc:       [N]      -- accumulated gradient for b (float32)
    // pre_x_shared:  [N]      -- current timestep pre_x
    // z_shared:      [N]      -- current timestep z
    // d_out_shared:  [N]      -- current timestep d_output
    // d_raw_shared:  [N]      -- d_raw for current timestep
    float* W_h_shared    = shared_mem;
    float* b_shared      = W_h_shared + N * N;
    float* h_prev_shared = b_shared + N;
    float* h_t_shared    = h_prev_shared + N;
    float* d_h_shared    = h_t_shared + N;
    float* d_W_h_acc     = d_h_shared + N;
    float* d_b_acc       = d_W_h_acc + N * N;
    float* pre_x_shared  = d_b_acc + N;
    float* z_shared      = pre_x_shared + N;
    float* d_out_shared  = z_shared + N;
    float* d_raw_shared  = d_out_shared + N;

    // Load W_h for this head (constant)
    int wh_offset = h * N * N;
    for (int i = tid; i < N * N; i += blockDim.x) {
        W_h_shared[i] = __bfloat162float(W_h[wh_offset + i]);
    }

    // Load bias for this head (needed for forward replay)
    if (tid < N) {
        b_shared[tid] = __bfloat162float(b_h[h * N + tid]);
    }

    // Initialize d_h to zero
    if (tid < N) {
        d_h_shared[tid] = 0.0f;
    }

    // Initialize d_W_h accumulator to zero
    for (int i = tid; i < N * N; i += blockDim.x) {
        d_W_h_acc[i] = 0.0f;
    }

    // Initialize d_b accumulator to zero
    if (tid < N) {
        d_b_acc[tid] = 0.0f;
    }
    __syncthreads();

    // Segment cache for h_{t-1} states
    // Layout: [B*H, checkpoint_interval, N]
    __nv_bfloat16* seg_cache_base = segment_cache + (size_t)block_idx * checkpoint_interval * N;

    int num_segments = (T + checkpoint_interval - 1) / checkpoint_interval;

    for (int seg = num_segments - 1; seg >= 0; seg--) {
        int t_start = seg * checkpoint_interval;
        int t_end = min(t_start + checkpoint_interval, T);
        int seg_len = t_end - t_start;

        // ================================================================
        // PHASE 1: Forward replay through segment, caching h_{t-1}
        // ================================================================

        // Load checkpoint for this segment into h_prev_shared
        // (checkpoint at index seg holds h state at the start of the segment)
        int cp_offset = (seg * B * H + b * H + h) * N;
        if (tid < N) {
            h_prev_shared[tid] = __bfloat162float(h_checkpoints[cp_offset + tid]);
        }
        __syncthreads();

        for (int local_t = 0; local_t < seg_len; local_t++) {
            int t = t_start + local_t;

            // Save h_{t-1} to segment cache BEFORE the update
            __nv_bfloat16* cache_slot = seg_cache_base + (size_t)local_t * N;
            if (tid < N) {
                cache_slot[tid] = __float2bfloat16(h_prev_shared[tid]);
            }
            __syncthreads();

            // Load pre_x for timestep t
            int io_offset = ((t * B + b) * H + h) * N;
            if (tid < N) {
                pre_x_shared[tid] = __bfloat162float(pre_x_all[io_offset + tid]);
            }
            __syncthreads();

            // Compute h_prev @ W_h and update h
            if (tid < N) {
                float wh_dot = 0.0f;
                #pragma unroll 8
                for (int i = 0; i < N; i++) {
                    wh_dot += h_prev_shared[i] * W_h_shared[i * N + tid];
                }
                float raw = pre_x_shared[tid] + wh_dot + b_shared[tid];
                h_prev_shared[tid] = tanhf(raw);
            }
            __syncthreads();
        }

        // ================================================================
        // PHASE 2: Backward pass through segment using cached h_{t-1}
        // ================================================================

        for (int local_t = seg_len - 1; local_t >= 0; local_t--) {
            int t = t_start + local_t;
            int io_offset = ((t * B + b) * H + h) * N;

            // Load cached h_{t-1}
            __nv_bfloat16* cache_slot = seg_cache_base + (size_t)local_t * N;
            if (tid < N) {
                h_prev_shared[tid] = __bfloat162float(cache_slot[tid]);
            }
            __syncthreads();

            // Load pre_x, z, d_output for timestep t
            if (tid < N) {
                pre_x_shared[tid] = __bfloat162float(pre_x_all[io_offset + tid]);
                z_shared[tid] = __bfloat162float(z_all[io_offset + tid]);
                d_out_shared[tid] = __bfloat162float(d_output[io_offset + tid]);
            }
            __syncthreads();

            // Recompute h_t = tanh(pre_x + h_prev @ W_h + b)
            if (tid < N) {
                float wh_dot = 0.0f;
                #pragma unroll 8
                for (int i = 0; i < N; i++) {
                    wh_dot += h_prev_shared[i] * W_h_shared[i * N + tid];
                }
                float raw = pre_x_shared[tid] + wh_dot + b_shared[tid];
                h_t_shared[tid] = tanhf(raw);
            }
            __syncthreads();

            // Backward through output gating: out = h_t * silu(z)
            //
            // d_h_from_out = d_out * silu(z)
            // d_z = d_out * h_t * dsilu(z)
            //   where dsilu(z) = sigmoid(z) * (1 + z * (1 - sigmoid(z)))
            //
            // Total d_h = d_h_from_out + d_h_from_future (already in d_h_shared)
            if (tid < N) {
                float z_val = z_shared[tid];
                float sig_z = 1.0f / (1.0f + expf(-z_val));
                float silu_z = z_val * sig_z;
                float dsilu_z = sig_z * (1.0f + z_val * (1.0f - sig_z));

                float d_out = d_out_shared[tid];
                float h_t = h_t_shared[tid];

                // d_z = d_out * h_t * dsilu(z)
                float d_z_val = d_out * h_t * dsilu_z;
                d_z_all[io_offset + tid] = __float2bfloat16(d_z_val);

                // Accumulate d_h: from output + from future timestep
                float d_h_total = d_out * silu_z + d_h_shared[tid];

                // Backward through tanh: d_raw = d_h_total * (1 - h_t^2)
                float dtanh = 1.0f - h_t * h_t;
                float d_raw = d_h_total * dtanh;

                // d_pre_x = d_raw
                d_pre_x_all[io_offset + tid] = __float2bfloat16(d_raw);

                // d_b += d_raw
                d_b_acc[tid] += d_raw;

                // Store d_raw for W_h gradient and h_prev gradient computation
                d_raw_shared[tid] = d_raw;
            }
            __syncthreads();

            // d_W_h[i,j] += h_prev[i] * d_raw[j] (outer product of h_prev and d_raw)
            // Thread tid computes row tid: d_W_h[tid, j] += h_prev[tid] * d_raw[j]
            if (tid < N) {
                #pragma unroll 8
                for (int j = 0; j < N; j++) {
                    d_W_h_acc[tid * N + j] += h_prev_shared[tid] * d_raw_shared[j];
                }
            }
            __syncthreads();

            // d_h_prev = W_h @ d_raw (propagate gradient to previous timestep)
            // For forward: raw[j] = sum_i h[i]*W_h[i,j], so d_h_prev[i] = sum_j W_h[i,j]*d_raw[j]
            if (tid < N) {
                float d_h_prev = 0.0f;
                #pragma unroll 8
                for (int j = 0; j < N; j++) {
                    d_h_prev += W_h_shared[tid * N + j] * d_raw_shared[j];
                }
                d_h_shared[tid] = d_h_prev;
            }
            __syncthreads();
        }
    }

    // Write accumulated d_W_h and d_b to global memory
    // d_W_h: [B, H, N, N] in float32
    int dwh_offset = ((b * H + h)) * N * N;
    for (int i = tid; i < N * N; i += blockDim.x) {
        d_W_h[dwh_offset + i] = d_W_h_acc[i];
    }

    // d_b: [B, H, N] in float32
    int db_offset = (b * H + h) * N;
    if (tid < N) {
        d_b_h[db_offset + tid] = d_b_acc[tid];
    }
}

// ============================================================================
// Dispatcher functions
// ============================================================================

void dispatch_e1h_forward(
    int T, int B, int H, int N,
    const __nv_bfloat16* pre_x_all,
    const __nv_bfloat16* z_all,
    __nv_bfloat16* h,
    const __nv_bfloat16* W_h,
    const __nv_bfloat16* b_h,
    __nv_bfloat16* output,
    __nv_bfloat16* h_checkpoints,
    int checkpoint_interval,
    cudaStream_t stream
) {
    // Shared memory: W_h(N*N) + b(N) + h(N) + pre_x(N) + z(N) = N*N + 4*N
    int shared_size = (N * N + 4 * N) * sizeof(float);
    int threads = N;  // One thread per state element
    int num_blocks = B * H;

    #define DISPATCH_E1H_FWD(N_VAL) do { \
        auto kernel = E1HForwardKernel_BF16<N_VAL>; \
        if (shared_size > 48 * 1024) { \
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); \
        } \
        kernel<<<num_blocks, N_VAL, shared_size, stream>>>( \
            T, B, H, pre_x_all, z_all, h, W_h, b_h, output, h_checkpoints, checkpoint_interval); \
    } while(0)

    if (N == 4) { DISPATCH_E1H_FWD(4); }
    else if (N == 8) { DISPATCH_E1H_FWD(8); }
    else if (N == 16) { DISPATCH_E1H_FWD(16); }
    else if (N == 24) { DISPATCH_E1H_FWD(24); }
    else if (N == 32) { DISPATCH_E1H_FWD(32); }
    else if (N == 36) { DISPATCH_E1H_FWD(36); }
    else if (N == 40) { DISPATCH_E1H_FWD(40); }
    else if (N == 44) { DISPATCH_E1H_FWD(44); }
    else if (N == 48) { DISPATCH_E1H_FWD(48); }
    else if (N == 56) { DISPATCH_E1H_FWD(56); }
    else if (N == 64) { DISPATCH_E1H_FWD(64); }
    else {
        fprintf(stderr, "E1H Forward: unsupported N=%d\n", N);
    }

    #undef DISPATCH_E1H_FWD
}

void dispatch_e1h_backward(
    int T, int B, int H, int N,
    const __nv_bfloat16* pre_x_all,
    const __nv_bfloat16* z_all,
    const __nv_bfloat16* W_h,
    const __nv_bfloat16* b_h,
    const __nv_bfloat16* h_checkpoints,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* d_pre_x_all,
    __nv_bfloat16* d_z_all,
    float* d_W_h,
    float* d_b_h,
    __nv_bfloat16* segment_cache,
    int checkpoint_interval,
    cudaStream_t stream
) {
    // Shared memory: W_h(N*N) + b(N) + h_prev(N) + h_t(N) + d_h(N)
    //              + d_W_h_acc(N*N) + d_b_acc(N) + pre_x(N) + z(N) + d_out(N) + d_raw(N)
    // Total: 2*N*N + 8*N
    int shared_size = (2 * N * N + 8 * N) * sizeof(float);
    int threads = N;  // One thread per state element
    int num_blocks = B * H;

    #define DISPATCH_E1H_BWD(N_VAL) do { \
        auto kernel = E1HBackwardKernel_BF16<N_VAL>; \
        if (shared_size > 48 * 1024) { \
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); \
        } \
        kernel<<<num_blocks, N_VAL, shared_size, stream>>>( \
            T, B, H, pre_x_all, z_all, W_h, b_h, h_checkpoints, d_output, \
            d_pre_x_all, d_z_all, d_W_h, d_b_h, segment_cache, checkpoint_interval); \
    } while(0)

    if (N == 4) { DISPATCH_E1H_BWD(4); }
    else if (N == 8) { DISPATCH_E1H_BWD(8); }
    else if (N == 16) { DISPATCH_E1H_BWD(16); }
    else if (N == 24) { DISPATCH_E1H_BWD(24); }
    else if (N == 32) { DISPATCH_E1H_BWD(32); }
    else if (N == 36) { DISPATCH_E1H_BWD(36); }
    else if (N == 40) { DISPATCH_E1H_BWD(40); }
    else if (N == 44) { DISPATCH_E1H_BWD(44); }
    else if (N == 48) { DISPATCH_E1H_BWD(48); }
    else if (N == 56) { DISPATCH_E1H_BWD(56); }
    else if (N == 64) { DISPATCH_E1H_BWD(64); }
    else {
        fprintf(stderr, "E1H Backward: unsupported N=%d\n", N);
    }

    #undef DISPATCH_E1H_BWD
}

// ============================================================================
// E1HForward Implementation
// ============================================================================

template<typename DataT>
E1HForward<DataT>::E1HForward(
    bool training,
    int batch_size,
    int n_state,
    int n_heads,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      n_state_(n_state),
      n_heads_(n_heads),
      stream_(stream) {}

template<typename DataT>
void E1HForward<DataT>::Run(
    int steps,
    const DataT* pre_x,    // [T, B, H, N]
    const DataT* z,         // [T, B, H, N]
    DataT* h,               // [B, H, N] initial state, updated in-place
    const DataT* W_h,       // [H, N, N]
    const DataT* b_h,       // [H, N]
    DataT* output,          // [T, B, H, N]
    DataT* h_cache          // checkpoint storage
) {
    int T = steps;
    int B = batch_size_;
    int N = n_state_;
    int H = n_heads_;

    dispatch_e1h_forward(
        T, B, H, N,
        (const __nv_bfloat16*)pre_x,
        (const __nv_bfloat16*)z,
        (__nv_bfloat16*)h,
        (const __nv_bfloat16*)W_h,
        (const __nv_bfloat16*)b_h,
        (__nv_bfloat16*)output,
        (__nv_bfloat16*)h_cache,
        E1H_CHECKPOINT_INTERVAL, stream_);
}

template struct E1HForward<__nv_bfloat16>;

// ============================================================================
// E1HBackward Implementation
// ============================================================================

template<typename DataT>
E1HBackward<DataT>::E1HBackward(
    int batch_size,
    int n_state,
    int n_heads,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      n_state_(n_state),
      n_heads_(n_heads),
      stream_(stream),
      segment_cache_(nullptr) {
    // Allocate segment cache: [B*H, checkpoint_interval, N]
    size_t seg_cache_size = (size_t)batch_size * n_heads * E1H_CHECKPOINT_INTERVAL * n_state * sizeof(DataT);
    cudaMalloc(&segment_cache_, seg_cache_size);
}

template<typename DataT>
E1HBackward<DataT>::~E1HBackward() {
    if (segment_cache_) {
        cudaFree(segment_cache_);
        segment_cache_ = nullptr;
    }
}

template<typename DataT>
void E1HBackward<DataT>::Run(
    int steps,
    const DataT* pre_x,        // [T, B, H, N]
    const DataT* z,             // [T, B, H, N]
    const DataT* W_h,           // [H, N, N]
    const DataT* b_h,           // [H, N]
    const DataT* h_checkpoints, // [num_checkpoints, B, H, N]
    const DataT* d_output,      // [T, B, H, N]
    DataT* d_pre_x,             // [T, B, H, N]
    DataT* d_z,                 // [T, B, H, N]
    float* d_W_h,               // [B, H, N, N] float32
    float* d_b_h                // [B, H, N] float32
) {
    int T = steps;
    int B = batch_size_;
    int N = n_state_;
    int H = n_heads_;

    dispatch_e1h_backward(
        T, B, H, N,
        (const __nv_bfloat16*)pre_x,
        (const __nv_bfloat16*)z,
        (const __nv_bfloat16*)W_h,
        (const __nv_bfloat16*)b_h,
        (const __nv_bfloat16*)h_checkpoints,
        (const __nv_bfloat16*)d_output,
        (__nv_bfloat16*)d_pre_x,
        (__nv_bfloat16*)d_z,
        d_W_h,
        d_b_h,
        (__nv_bfloat16*)segment_cache_,
        E1H_CHECKPOINT_INTERVAL, stream_);
}

template struct E1HBackward<__nv_bfloat16>;

}  // namespace elman

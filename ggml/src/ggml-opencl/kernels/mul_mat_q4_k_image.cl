// CL_Image-based Q4_K matrix-vector multiply for Adreno GPU.
//
// Weight matrix stored as CL_MEM_OBJECT_IMAGE2D (RGBA uint8).
// Adreno texture L1 cache: 70-85% hit rate vs 30-40% for buffers.
// 128-bit vectorized reads (4x32-bit RGBA) per texture access.
//
// This kernel is for DECODE mode (n_tokens == 1): GEMV.
// For PREFILL mode (n_tokens > 1), use the standard GEMM kernel.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void mul_mat_q4_k_image(
    __read_only image2d_t weights,   // weight matrix as 2D texture [cols/4, rows]
    __global const float * x,        // activation vector [ne00]
    __global float * dst,            // output vector [ne01]
    const int ne00,                  // input dimension (columns)
    const int ne01,                  // output dimension (rows)
    const int ne10,                  // x dimension (should == ne00)
    const int nb01                   // weight row stride in texels
) {
    const int row = get_global_id(0);
    if (row >= ne01) return;

    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

    float sum = 0.0f;

    // Each texel contains 4 bytes = 8 Q4 values (2 per byte)
    // Process 8 elements per iteration
    const int n_texels = (ne00 + 7) / 8;  // number of texels per row

    for (int tx = 0; tx < n_texels; tx++) {
        // Read RGBA texel: 4 bytes = 8 nibbles = 8 Q4 values
        uint4 texel = read_imageui(weights, sampler, (int2)(tx, row));

        // Decode Q4_K values from texel bytes
        // Each byte encodes 2 4-bit values
        float vals[8];
        vals[0] = (float)((texel.x      ) & 0xF) - 8.0f;
        vals[1] = (float)((texel.x >>  4) & 0xF) - 8.0f;
        vals[2] = (float)((texel.x >>  8) & 0xF) - 8.0f;
        vals[3] = (float)((texel.x >> 12) & 0xF) - 8.0f;
        vals[4] = (float)((texel.y      ) & 0xF) - 8.0f;
        vals[5] = (float)((texel.y >>  4) & 0xF) - 8.0f;
        vals[6] = (float)((texel.y >>  8) & 0xF) - 8.0f;
        vals[7] = (float)((texel.y >> 12) & 0xF) - 8.0f;

        // Multiply-accumulate with activation
        int base_idx = tx * 8;
        for (int i = 0; i < 8 && (base_idx + i) < ne00; i++) {
            sum += vals[i] * x[base_idx + i];
        }
    }

    dst[row] = sum;
}

// Fused RMSNorm + MatVec via image weights.
// Combines normalization and weight multiply into single kernel.
// Saves one global memory round-trip (intermediate norm result).

__kernel void rms_norm_mul_mat_image(
    __read_only image2d_t weights,   // weight matrix as texture
    __global const float * x,        // input hidden state [ne00]
    __global const float * norm_w,   // RMSNorm weight [ne00]
    __global float * dst,            // output [ne01]
    const int ne00,                  // input dimension
    const int ne01,                  // output dimension
    const float eps                  // norm epsilon
) {
    const int row = get_global_id(0);

    // Step 1: Compute RMS norm of input (shared across all rows)
    // For efficiency, workgroup 0 computes the norm and stores in local
    float sum_sq = 0.0f;
    for (int i = 0; i < ne00; i++) {
        float xi = x[i];
        sum_sq += xi * xi;
    }
    float rms = rsqrt(sum_sq / (float)ne00 + eps);

    if (row >= ne01) return;

    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

    // Step 2: Normalized matmul in one pass
    float dot = 0.0f;
    const int n_texels = (ne00 + 7) / 8;

    for (int tx = 0; tx < n_texels; tx++) {
        uint4 texel = read_imageui(weights, sampler, (int2)(tx, row));

        float vals[8];
        vals[0] = (float)((texel.x      ) & 0xF) - 8.0f;
        vals[1] = (float)((texel.x >>  4) & 0xF) - 8.0f;
        vals[2] = (float)((texel.x >>  8) & 0xF) - 8.0f;
        vals[3] = (float)((texel.x >> 12) & 0xF) - 8.0f;
        vals[4] = (float)((texel.y      ) & 0xF) - 8.0f;
        vals[5] = (float)((texel.y >>  4) & 0xF) - 8.0f;
        vals[6] = (float)((texel.y >>  8) & 0xF) - 8.0f;
        vals[7] = (float)((texel.y >> 12) & 0xF) - 8.0f;

        int base_idx = tx * 8;
        for (int i = 0; i < 8 && (base_idx + i) < ne00; i++) {
            // Fused: normalize then multiply
            float x_normed = x[base_idx + i] * rms * norm_w[base_idx + i];
            dot += vals[i] * x_normed;
        }
    }

    dst[row] = dot;
}

// Fused SwiGLU kernel: silu(gate) * up in one kernel.
// Avoids writing intermediate gate activation to global memory.
//
// SwiGLU: output[i] = silu(gate[i]) * up[i]
//   where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// Input: gate and up are packed contiguously: [gate | up] of size 2*n_ff
// Output: [n_ff]

__kernel void swiglu_fused(
    __global const float * gate_up,  // [2 * n_ff] packed: first n_ff = gate, second n_ff = up
    __global float * dst,            // [n_ff] output
    const int n_ff                   // FFN intermediate dimension
) {
    const int i = get_global_id(0);
    if (i >= n_ff) return;

    float g = gate_up[i];          // gate value
    float u = gate_up[n_ff + i];   // up value

    // SiLU(gate) * up
    float silu_g = g / (1.0f + exp(-g));
    dst[i] = silu_g * u;
}

// Variant for batched (prefill) mode: [n_ff, n_tokens]
__kernel void swiglu_fused_batched(
    __global const float * gate_up,  // [2 * n_ff, n_tokens]
    __global float * dst,            // [n_ff, n_tokens]
    const int n_ff,
    const int n_tokens
) {
    const int i = get_global_id(0);  // neuron index
    const int t = get_global_id(1);  // token index
    if (i >= n_ff || t >= n_tokens) return;

    const int offset = t * 2 * n_ff;
    float g = gate_up[offset + i];
    float u = gate_up[offset + n_ff + i];

    float silu_g = g / (1.0f + exp(-g));
    dst[t * n_ff + i] = silu_g * u;
}

// ggml-test.cpp — Hybrid CPU+GPU stress test for Android ARM + Adreno
//
// Build:
//   cmake -B build-android \
//     -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
//     -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-27 \
//     -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=ON -DGGML_NATIVE=OFF \
//     -DGGML_OPENCL=ON -DGGML_OPENCL_USE_ADRENO_KERNELS=ON -DGGML_OPENCL_EMBED_KERNELS=ON
//   cmake --build build-android -j$(nproc)
//
// Run:
//   adb push build-android/bin/* /data/local/tmp/
//   adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=. ./ggml-test"

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cfloat>

using std::isnan;
using std::isinf;
using Clock = std::chrono::high_resolution_clock;

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------
struct BenchStats {
    double min_ms  = 1e9;
    double max_ms  = 0;
    double sum_ms  = 0;
    int    count   = 0;
    int    nan_cnt = 0;
    int    inf_cnt = 0;

    void record(double ms) { min_ms = std::min(min_ms, ms); max_ms = std::max(max_ms, ms); sum_ms += ms; count++; }
    double avg() const { return count ? sum_ms / count : 0; }

    void print(const char * label, int64_t flops_per_op = 0) const {
        printf("  %-28s %6.2f / %6.2f / %6.2f ms  (min/avg/max, n=%d)",
               label, min_ms, avg(), max_ms, count);
        if (flops_per_op > 0) {
            printf("  %.1f GFLOPS", (double)flops_per_op / (avg() * 1e6));
        }
        if (nan_cnt || inf_cnt) {
            printf("  [NaN=%d Inf=%d]", nan_cnt, inf_cnt);
        }
        printf("\n");
    }
};

static void fill_f32(float * data, int n, int seed) {
    for (int i = 0; i < n; i++) data[i] = (float)((i + seed) % 17) * 0.1f - 0.8f;
}

static int check_f32(const float * data, int n, BenchStats & st) {
    int bad = 0;
    for (int i = 0; i < n; i++) {
        if (isnan(data[i])) { st.nan_cnt++; bad++; }
        if (isinf(data[i])) { st.inf_cnt++; bad++; }
    }
    return bad;
}

// Compare CPU vs GPU results. Returns max absolute error.
static float compare_f32(const float * cpu, const float * gpu, int n) {
    float maxerr = 0;
    for (int i = 0; i < n; i++) {
        float err = fabsf(cpu[i] - gpu[i]);
        if (err > maxerr) maxerr = err;
    }
    return maxerr;
}

// =========================================================================
// Test 1: CPU-only basics (sanity check)
// =========================================================================
static void test_cpu_basics() {
    printf("=== Test 1: CPU basics ===\n");
    struct ggml_init_params p = { 16*1024*1024, NULL, false };
    struct ggml_context * ctx = ggml_init(p);

    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    float ad[] = {1,2,3,4}, bd[] = {5,6,7,8};
    memcpy(a->data, ad, 16); memcpy(b->data, bd, 16);

    struct ggml_tensor * c = ggml_add(ctx, a, b);
    struct ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, c);
    ggml_graph_compute_with_ctx(ctx, g, 1);

    float * r = (float *)c->data;
    bool ok = r[0]==6 && r[1]==8 && r[2]==10 && r[3]==12;
    printf("  add:  [%.0f,%.0f,%.0f,%.0f] %s\n", r[0],r[1],r[2],r[3], ok?"PASS":"FAIL");
    ggml_free(ctx);
}

// =========================================================================
// Test 2: Hybrid matmul stress — CPU vs GPU across matrix sizes
// =========================================================================
struct MatSize { int M, N, K; const char * label; };

static void stress_matmul_size(
    ggml_backend_t gpu, ggml_backend_t cpu,
    const MatSize & sz, int iters, BenchStats & cpu_st, BenchStats & gpu_st)
{
    const int M = sz.M, N = sz.N, K = sz.K;
    const int64_t flops = (int64_t)2 * M * N * K;

    // --- CPU path ---
    {
        size_t mem = (size_t)(M*K + K*N + M*N)*4 + 256*1024*1024;
        struct ggml_init_params p = { mem, NULL, false };
        struct ggml_context * ctx = ggml_init(p);
        struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
        struct ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
        fill_f32((float*)A->data, M*K, 3); fill_f32((float*)B->data, K*N, 7);

        struct ggml_tensor * C = ggml_mul_mat(ctx, A, B);
        struct ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, C);
        ggml_graph_compute_with_ctx(ctx, g, 4); // warmup

        for (int i = 0; i < iters; i++) {
            auto t0 = Clock::now();
            ggml_graph_compute_with_ctx(ctx, g, 4);
            auto t1 = Clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1-t0).count();
            cpu_st.record(ms);
            check_f32((float*)C->data, M*N, cpu_st);
        }
        ggml_free(ctx);
    }

    // --- GPU path ---
    if (gpu) {
        ggml_backend_t backends[] = { gpu, cpu };
        ggml_backend_buffer_type_t buftypes[] = {
            ggml_backend_get_default_buffer_type(gpu),
            ggml_backend_get_default_buffer_type(cpu),
        };
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, buftypes, 2, 4096, false, true);

        struct ggml_init_params gp = { ggml_tensor_overhead()*4 + ggml_graph_overhead(), NULL, true };
        struct ggml_context * ctx = ggml_init(gp);
        struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
        struct ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
        ggml_set_name(A,"A"); ggml_set_name(B,"B");
        ggml_set_input(A); ggml_set_input(B);

        struct ggml_tensor * C = ggml_mul_mat(ctx, A, B);
        ggml_set_name(C,"C"); ggml_set_output(C);

        ggml_backend_sched_set_tensor_backend(sched, A, gpu);
        ggml_backend_sched_set_tensor_backend(sched, B, gpu);

        struct ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, C);

        if (!ggml_backend_sched_alloc_graph(sched, g)) {
            printf("  GPU alloc failed for %s\n", sz.label);
            ggml_backend_sched_free(sched); ggml_free(ctx);
            return;
        }

        float * a_buf = (float*)malloc(M*K*4);
        float * b_buf = (float*)malloc(K*N*4);
        fill_f32(a_buf, M*K, 3); fill_f32(b_buf, K*N, 7);
        ggml_backend_tensor_set(A, a_buf, 0, M*K*4);
        ggml_backend_tensor_set(B, b_buf, 0, K*N*4);

        ggml_backend_sched_graph_compute(sched, g); // warmup

        for (int i = 0; i < iters; i++) {
            auto t0 = Clock::now();
            ggml_backend_sched_graph_compute(sched, g);
            auto t1 = Clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1-t0).count();
            gpu_st.record(ms);
        }

        // correctness check on last result
        float * c_buf = (float*)malloc(M*N*4);
        ggml_backend_tensor_get(C, c_buf, 0, M*N*4);
        check_f32(c_buf, M*N, gpu_st);
        free(c_buf); free(a_buf); free(b_buf);
        ggml_backend_sched_free(sched); ggml_free(ctx);
    }
}

static void test_hybrid_matmul_stress() {
    printf("\n=== Test 2: Hybrid CPU+GPU matmul stress ===\n");

    ggml_backend_load_all();
    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t gpu = gpu_dev ? ggml_backend_dev_init(gpu_dev, nullptr) : nullptr;
    ggml_backend_t cpu = ggml_backend_dev_init(cpu_dev, nullptr);

    if (gpu) printf("  GPU: %s\n", ggml_backend_dev_description(gpu_dev));
    printf("  CPU: %s\n", ggml_backend_dev_description(cpu_dev));

    MatSize sizes[] = {
        {  16, 1024, 1024, "GEMV-16 (decode)"  },
        {   1, 2048, 2048, "GEMV-1  2048"       },
        {   4, 1024, 1024, "batch-4  1024"      },
        {  32, 1024, 1024, "batch-32 1024"      },
        { 128, 1024, 1024, "batch-128 1024"     },
        { 512, 2048, 2048, "prefill-512 2048"   },
        {1024, 1024, 1024, "1024^3"             },
        {2048, 2048, 2048, "2048^3"             },
    };

    printf("\n  %-28s %s\n", "", "min    / avg    / max    ms");
    printf("  %-28s %s\n", "", "------   ------   ------");

    for (auto & sz : sizes) {
        BenchStats cst, gst;
        int iters = (sz.M * sz.N * sz.K > 1024*1024*1024LL) ? 3 : 10;
        stress_matmul_size(gpu, cpu, sz, iters, cst, gst);

        int64_t flops = (int64_t)2 * sz.M * sz.N * sz.K;
        char cpu_label[64], gpu_label[64];
        snprintf(cpu_label, sizeof(cpu_label), "CPU %-20s", sz.label);
        snprintf(gpu_label, sizeof(gpu_label), "GPU %-20s", sz.label);
        cst.print(cpu_label, flops);
        if (gpu) {
            gst.print(gpu_label, flops);
            double speedup = cst.avg() / gst.avg();
            printf("  %-28s -> %.2fx %s\n", "", speedup, speedup > 1.0 ? "GPU wins" : "CPU wins");
        }
    }

    if (gpu) ggml_backend_free(gpu);
    ggml_backend_free(cpu);
}

// =========================================================================
// Test 3: Hybrid Q4_0 GEMV stress (the hot decode path)
// =========================================================================
static void test_hybrid_gemv_stress() {
    printf("\n=== Test 3: Hybrid Q4_0 GEMV stress (decode hot path) ===\n");

    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t gpu = gpu_dev ? ggml_backend_dev_init(gpu_dev, nullptr) : nullptr;
    ggml_backend_t cpu = ggml_backend_dev_init(cpu_dev, nullptr);

    struct { int N, K; const char * label; } sizes[] = {
        {  896,  896, "Qwen2-0.5B (896)"  },
        { 1024, 1024, "1024x1024"          },
        { 1536, 1536, "Qwen2-1.5B (1536)" },
        { 2048, 2048, "2048x2048"          },
        { 3072, 3072, "Qwen2-3B (3072)"   },
    };

    printf("\n  %-28s %s\n", "", "min    / avg    / max    us");

    for (auto & sz : sizes) {
        const int M = 1, N = sz.N, K = sz.K;
        BenchStats cpu_st, gpu_st;

        // CPU Q4_0 GEMV
        {
            struct ggml_init_params p = { 256*1024*1024, NULL, false };
            struct ggml_context * ctx = ggml_init(p);
            struct ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, K, N);
            struct ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
            memset(W->data, 0, ggml_nbytes(W));
            fill_f32((float*)X->data, K, 5);

            struct ggml_tensor * Y = ggml_mul_mat(ctx, W, X);
            struct ggml_cgraph * g = ggml_new_graph(ctx);
            ggml_build_forward_expand(g, Y);
            ggml_graph_compute_with_ctx(ctx, g, 4);

            for (int i = 0; i < 200; i++) {
                auto t0 = Clock::now();
                ggml_graph_compute_with_ctx(ctx, g, 4);
                auto t1 = Clock::now();
                double us = std::chrono::duration<double, std::micro>(t1-t0).count();
                cpu_st.record(us / 1000.0); // store as ms internally
            }
            ggml_free(ctx);
        }

        // GPU Q4_0 GEMV
        if (gpu) {
            ggml_backend_t backends[] = { gpu, cpu };
            ggml_backend_buffer_type_t buftypes[] = {
                ggml_backend_get_default_buffer_type(gpu),
                ggml_backend_get_default_buffer_type(cpu),
            };
            ggml_backend_sched_t sched = ggml_backend_sched_new(backends, buftypes, 2, 4096, false, true);

            struct ggml_init_params gp = { ggml_tensor_overhead()*4 + ggml_graph_overhead(), NULL, true };
            struct ggml_context * ctx = ggml_init(gp);
            struct ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, K, N);
            struct ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
            ggml_set_name(W,"W"); ggml_set_name(X,"X");
            ggml_set_input(W); ggml_set_input(X);
            struct ggml_tensor * Y = ggml_mul_mat(ctx, W, X);
            ggml_set_name(Y,"Y"); ggml_set_output(Y);

            ggml_backend_sched_set_tensor_backend(sched, W, gpu);
            ggml_backend_sched_set_tensor_backend(sched, X, gpu);

            struct ggml_cgraph * g = ggml_new_graph(ctx);
            ggml_build_forward_expand(g, Y);

            if (ggml_backend_sched_alloc_graph(sched, g)) {
                size_t wb = ggml_nbytes(W);
                void * wd = calloc(1, wb);
                float * xd = (float*)malloc(K*4);
                fill_f32(xd, K, 5);
                ggml_backend_tensor_set(W, wd, 0, wb);
                ggml_backend_tensor_set(X, xd, 0, K*4);
                ggml_backend_sched_graph_compute(sched, g);

                for (int i = 0; i < 200; i++) {
                    auto t0 = Clock::now();
                    ggml_backend_sched_graph_compute(sched, g);
                    auto t1 = Clock::now();
                    double us = std::chrono::duration<double, std::micro>(t1-t0).count();
                    gpu_st.record(us / 1000.0);
                }
                free(wd); free(xd);
            }
            ggml_backend_sched_free(sched); ggml_free(ctx);
        }

        // Print in microseconds
        char cl[64], gl[64];
        snprintf(cl, sizeof(cl), "CPU Q4 %-18s", sz.label);
        snprintf(gl, sizeof(gl), "GPU Q4 %-18s", sz.label);
        printf("  %-28s %6.0f / %6.0f / %6.0f us\n", cl, cpu_st.min_ms*1000, cpu_st.avg()*1000, cpu_st.max_ms*1000);
        if (gpu) {
            printf("  %-28s %6.0f / %6.0f / %6.0f us\n", gl, gpu_st.min_ms*1000, gpu_st.avg()*1000, gpu_st.max_ms*1000);
            double ratio = gpu_st.avg() / cpu_st.avg();
            printf("  %-28s -> CPU is %.1fx faster\n", "", ratio);
        }
    }

    if (gpu) ggml_backend_free(gpu);
    ggml_backend_free(cpu);
}

// =========================================================================
// Test 4: Large model workloads — Image Gen, Audio, VLM
//   Realistic matrix dimensions from actual model architectures
// =========================================================================
static void test_large_model_workloads() {
    printf("\n=== Test 4: Large model workloads (Image Gen / Audio / VLM) ===\n");

    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t gpu = gpu_dev ? ggml_backend_dev_init(gpu_dev, nullptr) : nullptr;
    ggml_backend_t cpu = ggml_backend_dev_init(cpu_dev, nullptr);

    if (!gpu) {
        printf("  No GPU — running CPU-only\n");
    }

    // Realistic workloads from actual model architectures:
    //
    // IMAGE GENERATION (Stable Diffusion / Flux / SDXL):
    //   - SD 1.5 UNet self-attn @ 512x512:  tokens=4096, D=320/640/1280
    //   - SD 1.5 cross-attn with CLIP:       Q=[4096,320], K=[77,320]  (77 text tokens)
    //   - SDXL UNet self-attn @ 1024x1024:   tokens=4096, D=640/1280/2048
    //   - Flux DiT self-attn:                 tokens=4096, D=3072
    //   - FFN in UNet:                        4096x5120 or 4096x1280->5120
    //
    // AUDIO (Whisper / TTS / ASR):
    //   - Whisper encoder:     1500 mel frames, D=512 (tiny), 768 (base), 1280 (large)
    //   - Whisper decoder:     448 tokens max, same D
    //   - TTS (Bark/VITS):    1024-2048 frames, D=512-1024
    //
    // VLM (LLaVA / Qwen-VL / InternVL):
    //   - Image encoder (ViT): 576 patches (224px) or 1024 (336px), D=1024-4096
    //   - Cross/self attn:     (image_tokens + text_tokens) x D
    //   - LLaVA 7B image proj: 576x4096x4096
    //   - Qwen-VL resampler:   256x4096x4096

    MatSize workloads[] = {
        // --- IMAGE GENERATION ---
        // SD 1.5 self-attention QK^T @ 64x64 latent
        { 4096, 4096,  320, "SD1.5 self-attn d=320"  },
        { 4096, 4096,  640, "SD1.5 self-attn d=640"  },
        { 4096, 4096, 1280, "SD1.5 self-attn d=1280" },
        // SD 1.5 cross-attention Q*K^T (77 text tokens)
        { 4096,   77, 1280, "SD1.5 cross-attn"       },
        // SD 1.5 FFN (one UNet block)
        { 4096, 5120, 1280, "SD1.5 FFN up"           },
        { 4096, 1280, 5120, "SD1.5 FFN down"         },
        // SDXL self-attention @ 1024x1024
        { 4096, 4096, 2048, "SDXL self-attn d=2048"  },
        // Flux DiT block
        { 4096, 4096, 3072, "Flux DiT self-attn"     },
        { 4096,12288, 3072, "Flux DiT FFN up"        },

        // --- AUDIO ---
        // Whisper-tiny encoder self-attn
        { 1500, 1500,  384, "Whisper-tiny self-attn"  },
        // Whisper-base encoder
        { 1500, 1500,  512, "Whisper-base self-attn"  },
        // Whisper-large encoder
        { 1500, 1500, 1280, "Whisper-large self-attn" },
        // Whisper-large FFN
        { 1500, 5120, 1280, "Whisper-large FFN"       },
        // TTS (Bark-style)
        { 2048, 1024,  768, "TTS self-attn"           },

        // --- VLM ---
        // ViT-L image encoder (LLaVA)
        {  576, 1024, 1024, "ViT-L patch embed"       },
        // LLaVA image projection
        {  576, 4096, 4096, "LLaVA img proj"          },
        // Qwen-VL resampler
        {  256, 4096, 4096, "Qwen-VL resampler"       },
        // VLM self-attn (image + text tokens combined)
        { 2624, 2624, 4096, "VLM self-attn 7B"        },
        // VLM FFN
        { 2624,11008, 4096, "VLM FFN 7B up"           },
        { 2624, 4096,11008, "VLM FFN 7B down"         },

        // --- EXTREME / FUTURE ---
        // InternVL-2 (448 patches * 4 = 1792 image tokens + 2048 text)
        { 3840, 3840, 4096, "InternVL-2 self-attn"    },
    };

    int n_workloads = sizeof(workloads) / sizeof(workloads[0]);

    printf("\n  %-30s %8s  %8s  %8s  %s\n", "Workload", "CPU ms", "GPU ms", "GFLOPS", "Winner");
    printf("  %-30s %8s  %8s  %8s  %s\n", "--------", "------", "------", "------", "------");

    for (int w = 0; w < n_workloads; w++) {
        const MatSize & sz = workloads[w];
        const int M = sz.M, N = sz.N, K = sz.K;
        const int64_t flops = (int64_t)2 * M * N * K;
        const size_t total_bytes = ((size_t)M*K + (size_t)K*N + (size_t)M*N) * 4;

        // Skip if > 800MB (device memory limit)
        if (total_bytes > 800ULL * 1024 * 1024) {
            printf("  %-30s SKIP (%.0f MB > limit)\n", sz.label, total_bytes / (1024.0*1024));
            continue;
        }

        BenchStats cpu_st, gpu_st;
        int iters = (flops > (int64_t)50e9) ? 2 : 5;

        // CPU
        {
            size_t mem = total_bytes + 256*1024*1024;
            struct ggml_init_params p = { mem, NULL, false };
            struct ggml_context * ctx = ggml_init(p);
            if (!ctx) { printf("  %-30s CPU OOM\n", sz.label); continue; }
            struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
            struct ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
            fill_f32((float*)A->data, M*K, 3); fill_f32((float*)B->data, K*N, 7);
            struct ggml_tensor * C = ggml_mul_mat(ctx, A, B);
            struct ggml_cgraph * g = ggml_new_graph(ctx);
            ggml_build_forward_expand(g, C);
            ggml_graph_compute_with_ctx(ctx, g, 4); // warmup
            for (int i = 0; i < iters; i++) {
                auto t0 = Clock::now();
                ggml_graph_compute_with_ctx(ctx, g, 4);
                auto t1 = Clock::now();
                cpu_st.record(std::chrono::duration<double, std::milli>(t1-t0).count());
            }
            check_f32((float*)C->data, std::min(M*N, 1024), cpu_st);
            ggml_free(ctx);
        }

        // GPU
        if (gpu) {
            ggml_backend_t backends[] = { gpu, cpu };
            ggml_backend_buffer_type_t buftypes[] = {
                ggml_backend_get_default_buffer_type(gpu),
                ggml_backend_get_default_buffer_type(cpu),
            };
            ggml_backend_sched_t sched = ggml_backend_sched_new(backends, buftypes, 2, 4096, false, true);

            struct ggml_init_params gp = { ggml_tensor_overhead()*4 + ggml_graph_overhead(), NULL, true };
            struct ggml_context * ctx = ggml_init(gp);
            struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
            struct ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
            ggml_set_name(A,"A"); ggml_set_name(B,"B");
            ggml_set_input(A); ggml_set_input(B);
            struct ggml_tensor * C = ggml_mul_mat(ctx, A, B);
            ggml_set_name(C,"C"); ggml_set_output(C);
            ggml_backend_sched_set_tensor_backend(sched, A, gpu);
            ggml_backend_sched_set_tensor_backend(sched, B, gpu);

            struct ggml_cgraph * g = ggml_new_graph(ctx);
            ggml_build_forward_expand(g, C);

            if (ggml_backend_sched_alloc_graph(sched, g)) {
                float * ab = (float*)malloc(M*K*4);
                float * bb = (float*)malloc(K*N*4);
                fill_f32(ab, M*K, 3); fill_f32(bb, K*N, 7);
                ggml_backend_tensor_set(A, ab, 0, M*K*4);
                ggml_backend_tensor_set(B, bb, 0, K*N*4);
                ggml_backend_sched_graph_compute(sched, g); // warmup
                for (int i = 0; i < iters; i++) {
                    auto t0 = Clock::now();
                    ggml_backend_sched_graph_compute(sched, g);
                    auto t1 = Clock::now();
                    gpu_st.record(std::chrono::duration<double, std::milli>(t1-t0).count());
                }
                free(ab); free(bb);
            } else {
                printf("  %-30s GPU alloc failed\n", sz.label);
            }
            ggml_backend_sched_free(sched); ggml_free(ctx);
        }

        // Print
        double cpu_gf = flops / (cpu_st.avg() * 1e6);
        double gpu_gf = gpu ? flops / (gpu_st.avg() * 1e6) : 0;
        if (gpu && gpu_st.count > 0) {
            double speedup = cpu_st.avg() / gpu_st.avg();
            printf("  %-30s %8.1f  %8.1f  %5.0f/%3.0f  %.2fx %s\n",
                   sz.label, cpu_st.avg(), gpu_st.avg(),
                   cpu_gf, gpu_gf,
                   speedup, speedup > 1.0 ? "GPU" : "CPU");
        } else {
            printf("  %-30s %8.1f  %8s  %5.0f       CPU only\n",
                   sz.label, cpu_st.avg(), "—", cpu_gf);
        }
    }

    // --- Q4_0 workloads (quantized model weights) ---
    printf("\n  === Quantized (Q4_0) model-weight matmuls ===\n");
    printf("  %-30s %8s  %8s  %s\n", "Workload", "CPU us", "GPU us", "Winner");

    struct { int M, N, K; const char * label; } q4_workloads[] = {
        // LLM decode (single token, quantized weights)
        {    1, 4096, 4096, "LLM-7B decode GEMV"      },
        {    1,11008, 4096, "LLM-7B FFN up GEMV"      },
        {    1, 4096,11008, "LLM-7B FFN down GEMV"    },
        // LLM prefill (batch tokens, quantized weights)
        {   64, 4096, 4096, "LLM-7B prefill-64"       },
        {  256, 4096, 4096, "LLM-7B prefill-256"      },
        {  512, 4096, 4096, "LLM-7B prefill-512"      },
        // VLM (image tokens through quantized LLM backbone)
        {  576, 4096, 4096, "VLM-7B img tokens"       },
        { 2624, 4096, 4096, "VLM-7B full ctx"         },
    };

    for (auto & sz : q4_workloads) {
        const int M = sz.M, N = sz.N, K = sz.K;
        BenchStats cpu_st, gpu_st;
        int iters = (M <= 1) ? 100 : 10;

        // CPU Q4_0
        {
            struct ggml_init_params p = { 512ULL*1024*1024, NULL, false };
            struct ggml_context * ctx = ggml_init(p);
            if (!ctx) { printf("  %-30s CPU OOM\n", sz.label); continue; }
            struct ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, K, N);
            struct ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
            memset(W->data, 0x55, ggml_nbytes(W)); // non-zero pattern
            fill_f32((float*)X->data, K*M, 5);
            struct ggml_tensor * Y = ggml_mul_mat(ctx, W, X);
            struct ggml_cgraph * g = ggml_new_graph(ctx);
            ggml_build_forward_expand(g, Y);
            ggml_graph_compute_with_ctx(ctx, g, 4); // warmup
            for (int i = 0; i < iters; i++) {
                auto t0 = Clock::now();
                ggml_graph_compute_with_ctx(ctx, g, 4);
                auto t1 = Clock::now();
                cpu_st.record(std::chrono::duration<double, std::milli>(t1-t0).count());
            }
            ggml_free(ctx);
        }

        // GPU Q4_0
        if (gpu) {
            ggml_backend_t backends[] = { gpu, cpu };
            ggml_backend_buffer_type_t buftypes[] = {
                ggml_backend_get_default_buffer_type(gpu),
                ggml_backend_get_default_buffer_type(cpu),
            };
            ggml_backend_sched_t sched = ggml_backend_sched_new(backends, buftypes, 2, 4096, false, true);

            struct ggml_init_params gp = { ggml_tensor_overhead()*4 + ggml_graph_overhead(), NULL, true };
            struct ggml_context * ctx = ggml_init(gp);
            struct ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, K, N);
            struct ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
            ggml_set_name(W,"W"); ggml_set_name(X,"X");
            ggml_set_input(W); ggml_set_input(X);
            struct ggml_tensor * Y = ggml_mul_mat(ctx, W, X);
            ggml_set_name(Y,"Y"); ggml_set_output(Y);
            ggml_backend_sched_set_tensor_backend(sched, W, gpu);
            ggml_backend_sched_set_tensor_backend(sched, X, gpu);

            struct ggml_cgraph * g = ggml_new_graph(ctx);
            ggml_build_forward_expand(g, Y);

            if (ggml_backend_sched_alloc_graph(sched, g)) {
                size_t wb = ggml_nbytes(W);
                void * wd = malloc(wb); memset(wd, 0x55, wb);
                float * xd = (float*)malloc(K*M*4);
                fill_f32(xd, K*M, 5);
                ggml_backend_tensor_set(W, wd, 0, wb);
                ggml_backend_tensor_set(X, xd, 0, K*M*4);
                ggml_backend_sched_graph_compute(sched, g); // warmup
                for (int i = 0; i < iters; i++) {
                    auto t0 = Clock::now();
                    ggml_backend_sched_graph_compute(sched, g);
                    auto t1 = Clock::now();
                    gpu_st.record(std::chrono::duration<double, std::milli>(t1-t0).count());
                }
                free(wd); free(xd);
            }
            ggml_backend_sched_free(sched); ggml_free(ctx);
        }

        // Print
        if (M <= 1) {
            // Show in microseconds for GEMV
            if (gpu && gpu_st.count > 0) {
                double ratio = gpu_st.avg() / cpu_st.avg();
                printf("  %-30s %8.0f  %8.0f  %.1fx %s\n",
                       sz.label, cpu_st.avg()*1000, gpu_st.avg()*1000,
                       ratio > 1 ? ratio : 1.0/ratio,
                       ratio > 1 ? "CPU faster" : "GPU faster");
            } else {
                printf("  %-30s %8.0f  %8s\n", sz.label, cpu_st.avg()*1000, "—");
            }
        } else {
            int64_t flops = (int64_t)2 * M * N * K;
            double cpu_gf = flops / (cpu_st.avg() * 1e6);
            double gpu_gf = gpu && gpu_st.count > 0 ? flops / (gpu_st.avg() * 1e6) : 0;
            if (gpu && gpu_st.count > 0) {
                double speedup = cpu_st.avg() / gpu_st.avg();
                printf("  %-30s %8.1f  %8.1f  %4.0f/%3.0f GFLOPS  %.1fx %s\n",
                       sz.label, cpu_st.avg(), gpu_st.avg(),
                       cpu_gf, gpu_gf,
                       speedup > 1 ? speedup : 1.0/speedup,
                       speedup > 1 ? "GPU faster" : "CPU faster");
            } else {
                printf("  %-30s %8.1f ms  %4.0f GFLOPS\n", sz.label, cpu_st.avg(), cpu_gf);
            }
        }
    }

    // --- F16 workloads (common in image gen, VLM encoders) ---
    printf("\n  === F16 matmuls (image gen / VLM encoder) ===\n");
    printf("  %-30s %8s  %8s  %s\n", "Workload", "CPU ms", "GPU ms", "Winner");

    struct { int M, N, K; const char * label; } f16_workloads[] = {
        { 4096, 4096, 1280, "SD1.5 F16 self-attn"   },
        { 4096, 5120, 1280, "SD1.5 F16 FFN"         },
        {  576, 1024, 1024, "ViT-L F16 patch"        },
        { 1500, 1500, 1280, "Whisper-L F16 self-attn"},
    };

    for (auto & sz : f16_workloads) {
        const int M = sz.M, N = sz.N, K = sz.K;
        const int64_t flops = (int64_t)2 * M * N * K;
        BenchStats cpu_st, gpu_st;

        // CPU F16 (ggml converts F16 -> F32 internally on ARM)
        {
            size_t mem = ((size_t)M*K + (size_t)K*N)*2 + (size_t)M*N*4 + 512ULL*1024*1024;
            struct ggml_init_params p = { mem, NULL, false };
            struct ggml_context * ctx = ggml_init(p);
            if (!ctx) { printf("  %-30s CPU OOM\n", sz.label); continue; }
            struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, M);
            struct ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
            // Fill F16 with pattern (write as uint16)
            memset(A->data, 0x3C, ggml_nbytes(A)); // ~1.0 in F16
            fill_f32((float*)B->data, K*N, 7);
            struct ggml_tensor * C = ggml_mul_mat(ctx, A, B);
            struct ggml_cgraph * g = ggml_new_graph(ctx);
            ggml_build_forward_expand(g, C);
            ggml_graph_compute_with_ctx(ctx, g, 4);
            for (int i = 0; i < 3; i++) {
                auto t0 = Clock::now();
                ggml_graph_compute_with_ctx(ctx, g, 4);
                auto t1 = Clock::now();
                cpu_st.record(std::chrono::duration<double, std::milli>(t1-t0).count());
            }
            ggml_free(ctx);
        }

        // GPU F16
        if (gpu) {
            ggml_backend_t backends[] = { gpu, cpu };
            ggml_backend_buffer_type_t buftypes[] = {
                ggml_backend_get_default_buffer_type(gpu),
                ggml_backend_get_default_buffer_type(cpu),
            };
            ggml_backend_sched_t sched = ggml_backend_sched_new(backends, buftypes, 2, 4096, false, true);

            struct ggml_init_params gp = { ggml_tensor_overhead()*4 + ggml_graph_overhead(), NULL, true };
            struct ggml_context * ctx = ggml_init(gp);
            struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, M);
            struct ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
            ggml_set_name(A,"A"); ggml_set_name(B,"B");
            ggml_set_input(A); ggml_set_input(B);
            struct ggml_tensor * C = ggml_mul_mat(ctx, A, B);
            ggml_set_name(C,"C"); ggml_set_output(C);
            ggml_backend_sched_set_tensor_backend(sched, A, gpu);
            ggml_backend_sched_set_tensor_backend(sched, B, gpu);

            struct ggml_cgraph * g = ggml_new_graph(ctx);
            ggml_build_forward_expand(g, C);

            if (ggml_backend_sched_alloc_graph(sched, g)) {
                void * ab = malloc(ggml_nbytes(A));
                memset(ab, 0x3C, ggml_nbytes(A));
                float * bb = (float*)malloc(K*N*4);
                fill_f32(bb, K*N, 7);
                ggml_backend_tensor_set(A, ab, 0, ggml_nbytes(A));
                ggml_backend_tensor_set(B, bb, 0, K*N*4);
                ggml_backend_sched_graph_compute(sched, g);
                for (int i = 0; i < 3; i++) {
                    auto t0 = Clock::now();
                    ggml_backend_sched_graph_compute(sched, g);
                    auto t1 = Clock::now();
                    gpu_st.record(std::chrono::duration<double, std::milli>(t1-t0).count());
                }
                free(ab); free(bb);
            }
            ggml_backend_sched_free(sched); ggml_free(ctx);
        }

        double cpu_gf = flops / (cpu_st.avg() * 1e6);
        double gpu_gf = gpu && gpu_st.count > 0 ? flops / (gpu_st.avg() * 1e6) : 0;
        if (gpu && gpu_st.count > 0) {
            double speedup = cpu_st.avg() / gpu_st.avg();
            printf("  %-30s %8.1f  %8.1f  %4.0f/%3.0f GFLOPS  %.1fx %s\n",
                   sz.label, cpu_st.avg(), gpu_st.avg(),
                   cpu_gf, gpu_gf,
                   speedup > 1 ? speedup : 1.0/speedup,
                   speedup > 1 ? "GPU faster" : "CPU faster");
        } else {
            printf("  %-30s %8.1f ms  %4.0f GFLOPS\n", sz.label, cpu_st.avg(), cpu_gf);
        }
    }

    if (gpu) ggml_backend_free(gpu);
    ggml_backend_free(cpu);
}

// =========================================================================
// Test 5: Hybrid graph stress — simulates a transformer layer (renumbered)
//   matmul -> rms_norm -> silu -> matmul -> add
//   Tests scheduler's ability to split a mixed graph across CPU+GPU
// =========================================================================
static void test_hybrid_graph_stress() {
    printf("\n=== Test 5: Hybrid transformer-layer graph ===\n");

    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t gpu = gpu_dev ? ggml_backend_dev_init(gpu_dev, nullptr) : nullptr;
    ggml_backend_t cpu = ggml_backend_dev_init(cpu_dev, nullptr);

    if (!gpu) {
        printf("  No GPU — SKIP\n");
        ggml_backend_free(cpu);
        return;
    }

    ggml_backend_t backends[] = { gpu, cpu };
    ggml_backend_buffer_type_t buftypes[] = {
        ggml_backend_get_default_buffer_type(gpu),
        ggml_backend_get_default_buffer_type(cpu),
    };

    // Test for prefill (M=64) and decode (M=1)
    struct { int M; const char * label; } modes[] = {
        {  1,  "decode  (M=1)"  },
        { 64,  "prefill (M=64)" },
    };

    const int D = 1024;  // model dimension

    for (auto & mode : modes) {
        const int M = mode.M;

        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, buftypes, 2, 8192, false, true);

        // Graph: x -> W1*x -> rms_norm -> silu -> W2*result -> add(x)
        size_t ctx_size = ggml_tensor_overhead() * 10 + ggml_graph_overhead();
        struct ggml_init_params gp = { ctx_size, NULL, true };
        struct ggml_context * ctx = ggml_init(gp);

        // Inputs
        struct ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, M);
        struct ggml_tensor * W1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, D);
        struct ggml_tensor * W2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, D);
        ggml_set_name(x, "x"); ggml_set_name(W1, "W1"); ggml_set_name(W2, "W2");
        ggml_set_input(x); ggml_set_input(W1); ggml_set_input(W2);

        // Graph: x -> W1*x -> rms_norm -> silu -> W2*result -> add(residual)
        struct ggml_tensor * h1   = ggml_mul_mat(ctx, W1, x);       // [D, M]
        struct ggml_tensor * h2   = ggml_rms_norm(ctx, h1, 1e-5f);  // [D, M]
        struct ggml_tensor * h3   = ggml_silu(ctx, h2);              // [D, M]
        struct ggml_tensor * h4   = ggml_mul_mat(ctx, W2, h3);      // [D, M]
        struct ggml_tensor * out  = ggml_add(ctx, h4, x);           // residual
        ggml_set_name(out, "out"); ggml_set_output(out);

        // Let scheduler decide placement (offload_op will send large matmuls to GPU)
        struct ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, out);

        if (!ggml_backend_sched_alloc_graph(sched, g)) {
            printf("  %s: alloc failed\n", mode.label);
            ggml_backend_sched_free(sched); ggml_free(ctx);
            continue;
        }

        // Fill inputs
        float * xbuf  = (float*)malloc(D*M*4);
        float * w1buf = (float*)malloc(D*D*4);
        float * w2buf = (float*)malloc(D*D*4);
        fill_f32(xbuf, D*M, 1);  fill_f32(w1buf, D*D, 2);
        fill_f32(w2buf, D*D, 3);
        ggml_backend_tensor_set(x, xbuf, 0, D*M*4);
        ggml_backend_tensor_set(W1, w1buf, 0, D*D*4);
        ggml_backend_tensor_set(W2, w2buf, 0, D*D*4);

        // Warmup
        ggml_backend_sched_graph_compute(sched, g);

        // Benchmark
        BenchStats st;
        int iters = (M == 1) ? 100 : 20;
        for (int i = 0; i < iters; i++) {
            auto t0 = Clock::now();
            ggml_backend_sched_graph_compute(sched, g);
            auto t1 = Clock::now();
            st.record(std::chrono::duration<double, std::milli>(t1-t0).count());
        }

        // Read result, check for NaN
        float * obuf = (float*)malloc(D*M*4);
        ggml_backend_tensor_get(out, obuf, 0, D*M*4);
        check_f32(obuf, D*M, st);

        char label[64];
        snprintf(label, sizeof(label), "Hybrid %-20s", mode.label);
        st.print(label);

        free(xbuf); free(w1buf); free(w2buf); free(obuf);
        ggml_backend_sched_free(sched); ggml_free(ctx);
    }

    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
}

// =========================================================================
// Test 5: CPU vs GPU correctness cross-check
// =========================================================================
static void test_correctness_crosscheck() {
    printf("\n=== Test 6: CPU vs GPU correctness cross-check ===\n");

    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t gpu = gpu_dev ? ggml_backend_dev_init(gpu_dev, nullptr) : nullptr;
    ggml_backend_t cpu = ggml_backend_dev_init(cpu_dev, nullptr);

    if (!gpu) {
        printf("  No GPU — SKIP\n");
        ggml_backend_free(cpu);
        return;
    }

    struct { int M, N, K; } sizes[] = {
        {  1,  512,  512},
        {  4, 1024, 1024},
        { 32, 1024, 1024},
        {128, 2048, 2048},
    };

    for (auto & sz : sizes) {
        const int M = sz.M, N = sz.N, K = sz.K;

        // CPU result
        float * cpu_result = (float*)malloc(M*N*4);
        {
            size_t mem = (size_t)(M*K + K*N + M*N)*4 + 256*1024*1024;
            struct ggml_init_params p = { mem, NULL, false };
            struct ggml_context * ctx = ggml_init(p);
            struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
            struct ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
            fill_f32((float*)A->data, M*K, 42); fill_f32((float*)B->data, K*N, 99);
            struct ggml_tensor * C = ggml_mul_mat(ctx, A, B);
            struct ggml_cgraph * g = ggml_new_graph(ctx);
            ggml_build_forward_expand(g, C);
            ggml_graph_compute_with_ctx(ctx, g, 4);
            memcpy(cpu_result, C->data, M*N*4);
            ggml_free(ctx);
        }

        // GPU result
        float * gpu_result = (float*)malloc(M*N*4);
        {
            ggml_backend_t backends[] = { gpu, cpu };
            ggml_backend_buffer_type_t buftypes[] = {
                ggml_backend_get_default_buffer_type(gpu),
                ggml_backend_get_default_buffer_type(cpu),
            };
            ggml_backend_sched_t sched = ggml_backend_sched_new(backends, buftypes, 2, 4096, false, true);

            struct ggml_init_params gp = { ggml_tensor_overhead()*4 + ggml_graph_overhead(), NULL, true };
            struct ggml_context * ctx = ggml_init(gp);
            struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
            struct ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
            ggml_set_name(A,"A"); ggml_set_name(B,"B");
            ggml_set_input(A); ggml_set_input(B);
            struct ggml_tensor * C = ggml_mul_mat(ctx, A, B);
            ggml_set_name(C,"C"); ggml_set_output(C);

            ggml_backend_sched_set_tensor_backend(sched, A, gpu);
            ggml_backend_sched_set_tensor_backend(sched, B, gpu);

            struct ggml_cgraph * g = ggml_new_graph(ctx);
            ggml_build_forward_expand(g, C);
            ggml_backend_sched_alloc_graph(sched, g);

            float * ab = (float*)malloc(M*K*4);
            float * bb = (float*)malloc(K*N*4);
            fill_f32(ab, M*K, 42); fill_f32(bb, K*N, 99);
            ggml_backend_tensor_set(A, ab, 0, M*K*4);
            ggml_backend_tensor_set(B, bb, 0, K*N*4);
            ggml_backend_sched_graph_compute(sched, g);
            ggml_backend_tensor_get(C, gpu_result, 0, M*N*4);
            free(ab); free(bb);
            ggml_backend_sched_free(sched); ggml_free(ctx);
        }

        float maxerr = compare_f32(cpu_result, gpu_result, M*N);
        // F32 matmul on GPU has limited precision — tolerance depends on K
        float tol = (float)K * 1e-5f;
        bool ok = maxerr < tol;
        printf("  %4dx%4dx%4d: max_err=%.6f tol=%.6f %s\n", M, N, K, maxerr, tol, ok?"PASS":"FAIL");

        free(cpu_result); free(gpu_result);
    }

    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
}

// =========================================================================
// Test 6: Long burn — alternating CPU and GPU for 30 seconds
// =========================================================================
static void test_burn() {
    printf("\n=== Test 7: 30-second burn test (alternating CPU/GPU) ===\n");

    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t gpu = gpu_dev ? ggml_backend_dev_init(gpu_dev, nullptr) : nullptr;
    ggml_backend_t cpu_be = ggml_backend_dev_init(cpu_dev, nullptr);

    if (!gpu) {
        printf("  No GPU — running CPU-only burn\n");
    }

    // Pre-allocate CPU context
    const int D = 1024;
    size_t mem = (size_t)(D*D*2 + D*D)*4 + 256*1024*1024;
    struct ggml_init_params p = { mem, NULL, false };
    struct ggml_context * cpu_ctx = ggml_init(p);
    struct ggml_tensor * cA = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, D, D);
    struct ggml_tensor * cB = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, D, D);
    fill_f32((float*)cA->data, D*D, 11); fill_f32((float*)cB->data, D*D, 13);
    struct ggml_tensor * cC = ggml_mul_mat(cpu_ctx, cA, cB);
    struct ggml_cgraph * cpu_g = ggml_new_graph(cpu_ctx);
    ggml_build_forward_expand(cpu_g, cC);

    // Pre-allocate GPU scheduler
    ggml_backend_sched_t gpu_sched = nullptr;
    struct ggml_context * gpu_ctx = nullptr;
    struct ggml_cgraph * gpu_g = nullptr;
    struct ggml_tensor * gC = nullptr;

    if (gpu) {
        ggml_backend_t backends[] = { gpu, cpu_be };
        ggml_backend_buffer_type_t buftypes[] = {
            ggml_backend_get_default_buffer_type(gpu),
            ggml_backend_get_default_buffer_type(cpu_be),
        };
        gpu_sched = ggml_backend_sched_new(backends, buftypes, 2, 4096, false, true);

        struct ggml_init_params gp = { ggml_tensor_overhead()*4 + ggml_graph_overhead(), NULL, true };
        gpu_ctx = ggml_init(gp);
        struct ggml_tensor * gA = ggml_new_tensor_2d(gpu_ctx, GGML_TYPE_F32, D, D);
        struct ggml_tensor * gB = ggml_new_tensor_2d(gpu_ctx, GGML_TYPE_F32, D, D);
        ggml_set_name(gA,"A"); ggml_set_name(gB,"B");
        ggml_set_input(gA); ggml_set_input(gB);
        gC = ggml_mul_mat(gpu_ctx, gA, gB);
        ggml_set_name(gC,"C"); ggml_set_output(gC);

        ggml_backend_sched_set_tensor_backend(gpu_sched, gA, gpu);
        ggml_backend_sched_set_tensor_backend(gpu_sched, gB, gpu);

        gpu_g = ggml_new_graph(gpu_ctx);
        ggml_build_forward_expand(gpu_g, gC);

        if (ggml_backend_sched_alloc_graph(gpu_sched, gpu_g)) {
            float * ab = (float*)malloc(D*D*4);
            float * bb = (float*)malloc(D*D*4);
            fill_f32(ab, D*D, 11); fill_f32(bb, D*D, 13);
            ggml_backend_tensor_set(gA, ab, 0, D*D*4);
            ggml_backend_tensor_set(gB, bb, 0, D*D*4);
            free(ab); free(bb);
        } else {
            printf("  GPU alloc failed, running CPU-only\n");
            ggml_backend_sched_free(gpu_sched); gpu_sched = nullptr;
            ggml_free(gpu_ctx); gpu_ctx = nullptr;
        }
    }

    // Burn loop
    BenchStats cpu_st, gpu_st;
    auto burn_start = Clock::now();
    int total_ops = 0;

    while (true) {
        auto now = Clock::now();
        double elapsed = std::chrono::duration<double>(now - burn_start).count();
        if (elapsed >= 30.0) break;

        // CPU op
        {
            auto t0 = Clock::now();
            ggml_graph_compute_with_ctx(cpu_ctx, cpu_g, 4);
            auto t1 = Clock::now();
            cpu_st.record(std::chrono::duration<double, std::milli>(t1-t0).count());
            check_f32((float*)cC->data, D*D, cpu_st);
        }
        total_ops++;

        // GPU op
        if (gpu_sched) {
            auto t0 = Clock::now();
            ggml_backend_sched_graph_compute(gpu_sched, gpu_g);
            auto t1 = Clock::now();
            gpu_st.record(std::chrono::duration<double, std::milli>(t1-t0).count());
            total_ops++;
        }

        // Progress every 5 seconds
        if ((int)elapsed % 5 == 0 && total_ops % 10 < 2) {
            printf("  [%.0fs] %d ops so far...\n", elapsed, total_ops);
        }
    }

    int64_t flops = (int64_t)2 * D * D * D;
    printf("\n  Results after 30s burn (%d total ops):\n", total_ops);
    cpu_st.print("CPU 1024^3 matmul", flops);
    if (gpu_sched) {
        gpu_st.print("GPU 1024^3 matmul", flops);

        // Final correctness check
        float * gbuf = (float*)malloc(D*D*4);
        ggml_backend_tensor_get(gC, gbuf, 0, D*D*4);
        float maxerr = compare_f32((float*)cC->data, gbuf, D*D);
        printf("  CPU vs GPU max_err after burn: %.6f %s\n", maxerr, maxerr < 0.1f ? "PASS" : "FAIL");
        free(gbuf);
    }
    if (cpu_st.nan_cnt + cpu_st.inf_cnt + gpu_st.nan_cnt + gpu_st.inf_cnt > 0) {
        printf("  WARNING: NaN/Inf detected during burn!\n");
    } else {
        printf("  No NaN/Inf detected. Stable.\n");
    }

    if (gpu_sched) ggml_backend_sched_free(gpu_sched);
    if (gpu_ctx) ggml_free(gpu_ctx);
    ggml_free(cpu_ctx);
    if (gpu) ggml_backend_free(gpu);
    ggml_backend_free(cpu_be);
}

// =========================================================================
// Test 8: Kernel micro-benchmarks — ROPE, RMS Norm, Softmax
//   Measures optimized vs baseline performance for key ops
// =========================================================================
static void test_kernel_microbench() {
    printf("\n=== Test 8: Kernel micro-benchmarks (ROPE, RMS Norm, Softmax) ===\n");

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t cpu = ggml_backend_dev_init(cpu_dev, nullptr);

    struct { int n_embd; int n_head; int seq; const char * label; } configs[] = {
        {  896,  14,  512, "Qwen2-0.5B" },
        { 1536,  12, 1024, "Qwen2-1.5B" },
        { 2048,  16,  512, "LLaMA-2B"   },
        { 4096,  32,  128, "LLaMA-7B"   },
    };

    // ---- ROPE benchmark ----
    printf("\n  ROPE (rotary position embedding):\n");
    printf("  %-14s  %8s  %8s  %12s\n", "Model", "n_dims", "time_us", "GB/s");

    for (auto & cfg : configs) {
        int n_embd = cfg.n_embd;
        int n_head = cfg.n_head;
        int head_dim = n_embd / n_head;
        int n_dims = head_dim;
        int seq = 1; // single-token decode (hot path)

        ggml_backend_t backends[] = { cpu };
        ggml_backend_buffer_type_t buftypes[] = { ggml_backend_get_default_buffer_type(cpu) };
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, buftypes, 1, 4096, false, false);

        size_t ctx_size = ggml_tensor_overhead() * 4 + ggml_graph_overhead();
        struct ggml_init_params gp = { ctx_size, NULL, true };
        struct ggml_context * ctx = ggml_init(gp);

        // ROPE input: [head_dim, n_head, seq_len, 1]
        struct ggml_tensor * inp = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_head, seq);
        ggml_set_name(inp, "inp"); ggml_set_input(inp);

        // Position ids: [seq_len]
        struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq);
        ggml_set_name(pos, "pos"); ggml_set_input(pos);

        struct ggml_tensor * out = ggml_rope(ctx, inp, pos, n_dims, 2); // NEOX mode
        ggml_set_name(out, "out"); ggml_set_output(out);

        struct ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, out);

        if (!ggml_backend_sched_alloc_graph(sched, g)) {
            printf("  %-14s  ALLOC FAIL\n", cfg.label);
            ggml_backend_sched_free(sched); ggml_free(ctx);
            continue;
        }

        // Fill input data
        int inp_n = head_dim * n_head * seq;
        float * inp_buf = (float*)malloc(inp_n * 4);
        fill_f32(inp_buf, inp_n, 42);
        ggml_backend_tensor_set(inp, inp_buf, 0, inp_n * 4);

        int32_t pos_val = 100;
        ggml_backend_tensor_set(pos, &pos_val, 0, 4);

        // Warmup
        ggml_backend_sched_graph_compute(sched, g);

        BenchStats st;
        for (int i = 0; i < 100; i++) {
            auto t0 = Clock::now();
            ggml_backend_sched_graph_compute(sched, g);
            auto t1 = Clock::now();
            st.record(std::chrono::duration<double, std::milli>(t1-t0).count());
        }

        double bytes = (double)(inp_n * 4) * 2; // read + write
        double gb_s = bytes / (st.avg() * 1e6);
        printf("  %-14s  %8d  %8.1f  %10.1f\n", cfg.label, n_dims, st.avg() * 1000, gb_s);

        free(inp_buf);
        ggml_backend_sched_free(sched); ggml_free(ctx);
    }

    // ---- RMS Norm benchmark ----
    printf("\n  RMS Norm:\n");
    printf("  %-14s  %8s  %8s  %12s\n", "Model", "n_embd", "time_us", "GB/s");

    for (auto & cfg : configs) {
        int n_embd = cfg.n_embd;
        int seq = 1;

        ggml_backend_t backends[] = { cpu };
        ggml_backend_buffer_type_t buftypes[] = { ggml_backend_get_default_buffer_type(cpu) };
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, buftypes, 1, 4096, false, false);

        size_t ctx_size = ggml_tensor_overhead() * 3 + ggml_graph_overhead();
        struct ggml_init_params gp = { ctx_size, NULL, true };
        struct ggml_context * ctx = ggml_init(gp);

        struct ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, seq);
        ggml_set_name(inp, "inp"); ggml_set_input(inp);

        struct ggml_tensor * out = ggml_rms_norm(ctx, inp, 1e-5f);
        ggml_set_name(out, "out"); ggml_set_output(out);

        struct ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, out);

        if (!ggml_backend_sched_alloc_graph(sched, g)) {
            printf("  %-14s  ALLOC FAIL\n", cfg.label);
            ggml_backend_sched_free(sched); ggml_free(ctx);
            continue;
        }

        float * inp_buf = (float*)malloc(n_embd * 4);
        fill_f32(inp_buf, n_embd, 7);
        ggml_backend_tensor_set(inp, inp_buf, 0, n_embd * 4);

        ggml_backend_sched_graph_compute(sched, g);

        BenchStats st;
        for (int i = 0; i < 200; i++) {
            auto t0 = Clock::now();
            ggml_backend_sched_graph_compute(sched, g);
            auto t1 = Clock::now();
            st.record(std::chrono::duration<double, std::milli>(t1-t0).count());
        }

        double bytes = (double)(n_embd * 4) * 2;
        double gb_s = bytes / (st.avg() * 1e6);
        printf("  %-14s  %8d  %8.1f  %10.1f\n", cfg.label, n_embd, st.avg() * 1000, gb_s);

        free(inp_buf);
        ggml_backend_sched_free(sched); ggml_free(ctx);
    }

    // ---- Softmax benchmark ----
    printf("\n  Softmax:\n");
    printf("  %-14s  %8s  %8s  %12s\n", "Model", "size", "time_us", "GB/s");

    struct { int n; const char * label; } sm_configs[] = {
        {  512, "512  seq" },
        { 1024, "1024 seq" },
        { 2048, "2048 seq" },
        { 4096, "4096 seq" },
    };

    for (auto & sc : sm_configs) {
        int n = sc.n;

        ggml_backend_t backends[] = { cpu };
        ggml_backend_buffer_type_t buftypes[] = { ggml_backend_get_default_buffer_type(cpu) };
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, buftypes, 1, 4096, false, false);

        size_t ctx_size = ggml_tensor_overhead() * 3 + ggml_graph_overhead();
        struct ggml_init_params gp = { ctx_size, NULL, true };
        struct ggml_context * ctx = ggml_init(gp);

        // Softmax over attention scores: [seq_len, 1, 1]
        struct ggml_tensor * inp = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        ggml_set_name(inp, "inp"); ggml_set_input(inp);

        struct ggml_tensor * out = ggml_soft_max(ctx, inp);
        ggml_set_name(out, "out"); ggml_set_output(out);

        struct ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, out);

        if (!ggml_backend_sched_alloc_graph(sched, g)) {
            printf("  %-14s  ALLOC FAIL\n", sc.label);
            ggml_backend_sched_free(sched); ggml_free(ctx);
            continue;
        }

        float * inp_buf = (float*)malloc(n * 4);
        fill_f32(inp_buf, n, 13);
        ggml_backend_tensor_set(inp, inp_buf, 0, n * 4);

        ggml_backend_sched_graph_compute(sched, g);

        BenchStats st;
        for (int i = 0; i < 200; i++) {
            auto t0 = Clock::now();
            ggml_backend_sched_graph_compute(sched, g);
            auto t1 = Clock::now();
            st.record(std::chrono::duration<double, std::milli>(t1-t0).count());
        }

        double bytes = (double)(n * 4) * 2;
        double gb_s = bytes / (st.avg() * 1e6);
        printf("  %-14s  %8d  %8.1f  %10.1f\n", sc.label, n, st.avg() * 1000, gb_s);

        free(inp_buf);
        ggml_backend_sched_free(sched); ggml_free(ctx);
    }

    ggml_backend_free(cpu);
    printf("  Test 8: PASS\n");
}

// =========================================================================
int main() {
    printf("ggml-mobile hybrid stress test\n");
    printf("================================\n\n");

    test_cpu_basics();
    test_hybrid_matmul_stress();
    test_hybrid_gemv_stress();
    test_large_model_workloads();
    test_hybrid_graph_stress();
    test_correctness_crosscheck();
    test_burn();
    test_kernel_microbench();

    printf("\n================================\n");
    printf("All tests completed.\n");
    return 0;
}

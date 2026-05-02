#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TN_THREAD_POWER_SAVING = 0,
    TN_THREAD_BALANCED     = 1,
    TN_THREAD_PERFORMANCE  = 2,
} tn_thread_mode;

typedef struct {
    int32_t  n_cores_total;
    int32_t  n_perf_cores;
    int32_t  n_efficiency_cores;
    int32_t  max_freq_khz;
    int32_t  min_freq_khz;
} tn_device_info;

typedef struct {
    int32_t  n_threads_generation;
    int32_t  n_threads_batch;
    int32_t  n_batch;
    bool     pin_to_perf_cores;
    int32_t  perf_core_ids[16];
    int32_t  n_perf_core_ids;
    int32_t  efficiency_core_ids[16];
    int32_t  n_efficiency_core_ids;
} tn_thread_config;

// Detect device topology (reads /sys/devices/system/cpu/ on Android/Linux)
tn_device_info   tn_detect_device(void);

// Get thread config for a given mode
tn_thread_config tn_thread_config_for_mode(tn_thread_mode mode);

// Get recommended batch size based on available memory
int32_t          tn_recommend_batch_size(int64_t model_size_bytes);

// Model sizing: max model bytes that fits in available_ram_bytes with room for KV + overhead
int64_t          tn_max_model_size(int64_t available_ram_bytes, int32_t n_ctx);

// Query available RAM on the device
int64_t          tn_available_ram_bytes(void);

#ifdef __cplusplus
}
#endif

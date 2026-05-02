#include "thread-engine.h"
#include "tn-log.h"

#include <cstdio>
#include <algorithm>

#ifdef __ANDROID__
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

#if defined(__linux__) || defined(__ANDROID__)
#include <dirent.h>
#endif

// Read integer from sysfs path, returns -1 on failure
static int read_sysfs_int(const char * path) {
    FILE * f = fopen(path, "r");
    if (!f) return -1;
    int val = -1;
    if (fscanf(f, "%d", &val) != 1) val = -1;
    fclose(f);
    return val;
}

// Read max frequency for a CPU core in KHz
static int core_max_freq_khz(int cpu_id) {
    char path[128];
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu_id);
    return read_sysfs_int(path);
}

// Check if a CPU core is online
static bool core_is_online(int cpu_id) {
    if (cpu_id == 0) return true; // cpu0 is always online
    char path[128];
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu%d/online", cpu_id);
    return read_sysfs_int(path) == 1;
}

tn_device_info tn_detect_device(void) {
    tn_device_info info = {};

#if defined(__linux__) || defined(__ANDROID__)
    // Count total cores
    DIR * dir = opendir("/sys/devices/system/cpu");
    if (!dir) {
        info.n_cores_total = 1;
        return info;
    }

    int freqs[64] = {};
    int n_cores = 0;

    struct dirent * entry;
    while ((entry = readdir(dir)) != nullptr) {
        int cpu_id = -1;
        if (sscanf(entry->d_name, "cpu%d", &cpu_id) == 1 && cpu_id >= 0 && cpu_id < 64) {
            if (!core_is_online(cpu_id)) continue;
            freqs[n_cores] = core_max_freq_khz(cpu_id);
            n_cores++;
        }
    }
    closedir(dir);

    if (n_cores == 0) {
        info.n_cores_total = 1;
        return info;
    }

    info.n_cores_total = n_cores;

    // Find max and min frequencies
    int max_freq = 0, min_freq = 0x7FFFFFFF;
    for (int i = 0; i < n_cores; i++) {
        if (freqs[i] > 0) {
            if (freqs[i] > max_freq) max_freq = freqs[i];
            if (freqs[i] < min_freq) min_freq = freqs[i];
        }
    }
    info.max_freq_khz = max_freq;
    info.min_freq_khz = min_freq;

    // Classify cores: anything above 70% of max is a performance core
    int threshold = max_freq > 0 ? (int)(max_freq * 0.70) : 0;
    int n_perf = 0, n_eff = 0;
    for (int i = 0; i < n_cores; i++) {
        if (freqs[i] >= threshold) n_perf++;
        else n_eff++;
    }

    info.n_perf_cores = n_perf;
    info.n_efficiency_cores = n_eff;

    TN_LOG_INF("device: %d cores (%d perf, %d eff), freq %d-%d MHz",
               n_cores, n_perf, n_eff, min_freq / 1000, max_freq / 1000);

#else
    // Fallback for non-Linux
    int n = 4;
#ifdef _SC_NPROCESSORS_ONLN
    n = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1) n = 4;
#endif
    info.n_cores_total = n;
    info.n_perf_cores = n;
    info.n_efficiency_cores = 0;
#endif

    return info;
}

tn_thread_config tn_thread_config_for_mode(tn_thread_mode mode) {
    tn_thread_config cfg = {};
    tn_device_info dev = tn_detect_device();

#if defined(__linux__) || defined(__ANDROID__)
    // Build sorted core lists by frequency
    struct core_info { int id; int freq; };
    core_info cores[64] = {};
    int n = 0;

    DIR * dir = opendir("/sys/devices/system/cpu");
    if (dir) {
        struct dirent * entry;
        while ((entry = readdir(dir)) != nullptr && n < 64) {
            int cpu_id = -1;
            if (sscanf(entry->d_name, "cpu%d", &cpu_id) == 1 && cpu_id >= 0) {
                if (!core_is_online(cpu_id)) continue;
                cores[n].id = cpu_id;
                cores[n].freq = core_max_freq_khz(cpu_id);
                n++;
            }
        }
        closedir(dir);
    }

    // Sort by frequency descending (fastest first)
    std::sort(cores, cores + n, [](const core_info & a, const core_info & b) {
        return a.freq > b.freq;
    });

    // Classify using 70% threshold
    int threshold = n > 0 ? (int)(cores[0].freq * 0.70) : 0;
    int n_perf = 0, n_eff = 0;
    for (int i = 0; i < n; i++) {
        if (cores[i].freq >= threshold && n_perf < 16) {
            cfg.perf_core_ids[n_perf++] = cores[i].id;
        } else if (n_eff < 16) {
            cfg.efficiency_core_ids[n_eff++] = cores[i].id;
        }
    }
    cfg.n_perf_core_ids = n_perf;
    cfg.n_efficiency_core_ids = n_eff;
#else
    cfg.n_perf_core_ids = dev.n_perf_cores;
    cfg.n_efficiency_core_ids = 0;
#endif

    int np = cfg.n_perf_core_ids > 0 ? cfg.n_perf_core_ids : dev.n_cores_total;
    int n_total = dev.n_cores_total > 0 ? dev.n_cores_total : 4;

    switch (mode) {
        case TN_THREAD_POWER_SAVING:
            // Use 1-2 efficiency cores, small batch
            cfg.n_threads_generation = 1;
            cfg.n_threads_batch = std::max(1, cfg.n_efficiency_core_ids > 0 ? cfg.n_efficiency_core_ids : 2);
            cfg.n_batch = 128;
            cfg.pin_to_perf_cores = false;
            break;

        case TN_THREAD_BALANCED:
            cfg.n_threads_generation = std::min(2, np);
            cfg.n_threads_batch = np;
            // Bumped from 256: VLM image-embedding prompt eval is dominated by
            // the number of llama_decode calls, each of which pays the graph
            // build / scheduler overhead. Doubling n_batch halves those calls
            // and measurably improves image-token prompt-eval throughput on
            // 350M–1B models. Extra KV-compute buffer cost is <100 MiB.
            cfg.n_batch = 512;
            cfg.pin_to_perf_cores = true;
            break;

        case TN_THREAD_PERFORMANCE:
            cfg.n_threads_generation = std::min(4, np);
            cfg.n_threads_batch = n_total;
            cfg.n_batch = 1024;
            cfg.pin_to_perf_cores = true;
            break;
    }

    TN_LOG_INF("thread mode %d: gen=%d batch=%d n_batch=%d pin=%d",
               (int)mode, cfg.n_threads_generation, cfg.n_threads_batch,
               cfg.n_batch, cfg.pin_to_perf_cores);

    return cfg;
}

int32_t tn_recommend_batch_size(int64_t model_size_bytes) {
    int64_t ram = tn_available_ram_bytes();
    if (ram <= 0) return 256; // safe default

    // Ratio of free RAM to model size determines batch budget
    double ratio = (double)ram / (double)(model_size_bytes > 0 ? model_size_bytes : 1);

    if (ratio < 1.5) return 64;   // very tight
    if (ratio < 2.0) return 128;  // tight
    if (ratio < 3.0) return 256;  // comfortable
    return 512;                    // plenty
}

int64_t tn_available_ram_bytes(void) {
#if defined(__ANDROID__) || defined(__linux__)
    FILE * f = fopen("/proc/meminfo", "r");
    if (!f) return -1;

    int64_t mem_available = -1;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        long long val = 0;
        if (sscanf(line, "MemAvailable: %lld kB", &val) == 1) {
            mem_available = (int64_t)val * 1024;
            break;
        }
    }
    fclose(f);
    return mem_available;
#else
    return -1;
#endif
}

int64_t tn_max_model_size(int64_t available_ram_bytes, int32_t n_ctx) {
    if (available_ram_bytes <= 0) return 0;

    // Reserve for KV cache: ~0.5 MB per 1024 ctx tokens (rough estimate for small models)
    int64_t kv_estimate = ((int64_t)n_ctx / 1024) * 512 * 1024;
    if (kv_estimate < 64 * 1024 * 1024) kv_estimate = 64 * 1024 * 1024; // min 64 MB

    // Reserve 200 MB for OS + app + scratch buffers
    int64_t overhead = 200LL * 1024 * 1024;

    int64_t budget = available_ram_bytes - kv_estimate - overhead;
    return budget > 0 ? budget : 0;
}

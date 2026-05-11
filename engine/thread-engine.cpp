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

// Cluster-based classification: groups cores by max frequency, treats the
// fastest cluster (and any within 5% of it) as "perf". Replaces the old
// 70%-of-max threshold which misclassified Cortex-A520 eff cores on
// Snapdragon 7s Gen 3 / 7+ Gen 3 as perf (eff @ 1.95 GHz / prime @ 2.5 GHz
// = 78%, above 70%, wrong cluster).
//
// "Perf" = top frequency cluster. "Eff" = everything else.
static int classify_perf_count(const int * freqs, int n) {
    if (n <= 0) return 0;
    int max_freq = 0;
    for (int i = 0; i < n; i++) if (freqs[i] > max_freq) max_freq = freqs[i];
    if (max_freq <= 0) return n;
    // 5% jitter tolerance — same-cluster cores can drift slightly.
    const int cluster_threshold = (int)(max_freq * 0.95);
    int n_perf = 0;
    for (int i = 0; i < n; i++) {
        if (freqs[i] >= cluster_threshold) n_perf++;
    }
    return n_perf;
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

    int max_freq = 0, min_freq = 0x7FFFFFFF;
    for (int i = 0; i < n_cores; i++) {
        if (freqs[i] > 0) {
            if (freqs[i] > max_freq) max_freq = freqs[i];
            if (freqs[i] < min_freq) min_freq = freqs[i];
        }
    }
    info.max_freq_khz = max_freq;
    info.min_freq_khz = min_freq;

    int n_perf = classify_perf_count(freqs, n_cores);
    int n_eff  = n_cores - n_perf;

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

    // Cluster-based: top frequency tier (within 5% of max) is "perf";
    // everything else is "eff". See classify_perf_count() for rationale.
    const int cluster_threshold = n > 0 ? (int)(cores[0].freq * 0.95) : 0;
    int n_perf = 0, n_eff = 0;
    for (int i = 0; i < n; i++) {
        if (cores[i].freq >= cluster_threshold && n_perf < 16) {
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

    // np = TRUE perf cluster count (post cluster-classification). On modern
    // big.LITTLE SoCs this is typically 4 (e.g. Snapdragon 7s Gen 3:
    // 4× A720 perf + 4× A520 eff). For BALANCED we keep work on the perf
    // cluster only — ggml's parallel kernels are pace-bound by the slowest
    // thread, and the eff cores are 25–40% slower per cycle PLUS share less
    // L2/L3 cache. Mixing them in BALANCED hurts. PERFORMANCE deliberately
    // pulls in eff cores — see comment in that case below.
    int np      = cfg.n_perf_core_ids > 0 ? cfg.n_perf_core_ids : dev.n_cores_total;
    int n_total = dev.n_cores_total > 0 ? dev.n_cores_total : 4;
    if (np <= 0) np = 4;

    switch (mode) {
        case TN_THREAD_POWER_SAVING:
            // 1 thread for decode, 2 for batch, run on eff cores (no pinning).
            cfg.n_threads_generation = 1;
            cfg.n_threads_batch = std::max(1,
                cfg.n_efficiency_core_ids > 0 ? std::min(2, cfg.n_efficiency_core_ids) : 2);
            cfg.n_batch = 128;
            cfg.pin_to_perf_cores = false;
            break;

        case TN_THREAD_BALANCED:
            // Decode is memory-bandwidth-bound + shared-L3 sensitive.
            // Two empirically-validated rules on Snapdragon 7s Gen 3:
            //   1. gen=2 beats gen=4. More threads thrash L3 — LFM-350M
            //      Q4_K_M went from 38 → 28 tk/s when we bumped to 4.
            //   2. pin=false beats pin-to-perf-cluster. The big.LITTLE-
            //      aware Android scheduler picks the natural prime + 1-perf
            //      pair for active decode threads; explicit affinity
            //      restricting to all 4 perf cores leaves the scheduler
            //      with worse placement choices and also occasionally
            //      EINVALs when the cpuset controller restricts the
            //      foreground service to a subset.
            cfg.n_threads_generation = std::min(2, np);
            cfg.n_threads_batch      = np;
            // Bumped from 256: VLM image-embedding prompt eval is dominated by
            // the number of llama_decode calls, each of which pays the graph
            // build / scheduler overhead. Doubling n_batch halves those calls
            // and measurably improves image-token prompt-eval throughput on
            // 350M–1B models. Extra KV-compute buffer cost is <100 MiB.
            cfg.n_batch = 512;
            cfg.pin_to_perf_cores = false;
            break;

        case TN_THREAD_PERFORMANCE:
            // PERFORMANCE differs from BALANCED only on the batch axis —
            // not the decode axis. Pure decode is memory-bandwidth-bound
            // and gen=2 is the sweet spot on shared-L3 perf clusters
            // regardless of mode (see BALANCED comment). Where PERFORMANCE
            // actually helps is prompt eval and image-prefill: those are
            // compute-bound so we widen the threadpool to every core
            // (incl. eff cluster — eff cores at 78% of perf still
            // contribute meaningfully to compute-bound parallel-for) and
            // bump n_batch to 1024 to halve graph-build overhead on long
            // prompts. No pin — kernel can spread across all 8 cores so
            // memory transactions on different cores overlap.
            cfg.n_threads_generation = std::min(2, np);
            cfg.n_threads_batch      = n_total;
            cfg.n_batch              = 1024;
            cfg.pin_to_perf_cores    = false;
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

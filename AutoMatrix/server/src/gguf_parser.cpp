#include "gguf_parser.h"
#include "gguf.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <sstream>
#include <vector>
#include <map>

// Simple JSON builder (no external dependency)
static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;
        }
    }
    return out;
}

static std::string kv_to_json_value(struct gguf_context * ctx, int64_t i) {
    enum gguf_type type = gguf_get_kv_type(ctx, i);
    switch (type) {
        case GGUF_TYPE_UINT8:   return std::to_string(gguf_get_val_u8(ctx, i));
        case GGUF_TYPE_INT8:    return std::to_string(gguf_get_val_i8(ctx, i));
        case GGUF_TYPE_UINT16:  return std::to_string(gguf_get_val_u16(ctx, i));
        case GGUF_TYPE_INT16:   return std::to_string(gguf_get_val_i16(ctx, i));
        case GGUF_TYPE_UINT32:  return std::to_string(gguf_get_val_u32(ctx, i));
        case GGUF_TYPE_INT32:   return std::to_string(gguf_get_val_i32(ctx, i));
        case GGUF_TYPE_UINT64:  return std::to_string(gguf_get_val_u64(ctx, i));
        case GGUF_TYPE_INT64:   return std::to_string(gguf_get_val_i64(ctx, i));
        case GGUF_TYPE_FLOAT32: {
            char buf[32];
            snprintf(buf, sizeof(buf), "%.6g", (double)gguf_get_val_f32(ctx, i));
            return buf;
        }
        case GGUF_TYPE_FLOAT64: {
            char buf[32];
            snprintf(buf, sizeof(buf), "%.6g", gguf_get_val_f64(ctx, i));
            return buf;
        }
        case GGUF_TYPE_BOOL:
            return gguf_get_val_bool(ctx, i) ? "true" : "false";
        case GGUF_TYPE_STRING:
            return "\"" + json_escape(gguf_get_val_str(ctx, i)) + "\"";
        case GGUF_TYPE_ARRAY: {
            // For arrays, just report type and count
            size_t n = gguf_get_arr_n(ctx, i);
            return "\"[array:" + std::to_string(n) + "]\"";
        }
        default:
            return "null";
    }
}

// Helper to get u32 with fallback
static uint32_t get_u32(struct gguf_context * ctx, const char * key, uint32_t fallback) {
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) return fallback;
    enum gguf_type t = gguf_get_kv_type(ctx, id);
    if (t == GGUF_TYPE_UINT32) return gguf_get_val_u32(ctx, id);
    if (t == GGUF_TYPE_INT32)  return (uint32_t)gguf_get_val_i32(ctx, id);
    if (t == GGUF_TYPE_UINT64) return (uint32_t)gguf_get_val_u64(ctx, id);
    if (t == GGUF_TYPE_INT64)  return (uint32_t)gguf_get_val_i64(ctx, id);
    return fallback;
}

static const char * get_str(struct gguf_context * ctx, const char * key, const char * fallback) {
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) return fallback;
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) return fallback;
    return gguf_get_val_str(ctx, id);
}

std::string parse_gguf(const std::string & filepath) {
    // Create ggml_context to get tensor shapes (no_alloc=true, no data)
    struct ggml_context * ggml_ctx = nullptr;
    struct gguf_init_params params = { true, &ggml_ctx };
    struct gguf_context * ctx = gguf_init_from_file(filepath.c_str(), params);
    if (!ctx) {
        return "";
    }

    // Extract architecture prefix
    std::string arch = get_str(ctx, "general.architecture", "unknown");
    std::string name = get_str(ctx, "general.name", "");

    // Model config
    uint32_t n_embd      = get_u32(ctx, (arch + ".embedding_length").c_str(), 0);
    uint32_t n_layer     = get_u32(ctx, (arch + ".block_count").c_str(), 0);
    uint32_t n_head      = get_u32(ctx, (arch + ".attention.head_count").c_str(), 0);
    uint32_t n_head_kv   = get_u32(ctx, (arch + ".attention.head_count_kv").c_str(), n_head);
    uint32_t n_ctx       = get_u32(ctx, (arch + ".context_length").c_str(), 0);
    uint32_t n_vocab     = get_u32(ctx, (arch + ".vocab_size").c_str(), 0);
    // Fallback vocab from tokenizer
    if (n_vocab == 0) {
        int64_t tok_id = gguf_find_key(ctx, "tokenizer.ggml.tokens");
        if (tok_id >= 0) n_vocab = (uint32_t)gguf_get_arr_n(ctx, tok_id);
    }

    // Tensor info
    int64_t n_tensors = gguf_get_n_tensors(ctx);

    // Determine primary quant type from a transformer block weight (not embedding)
    std::string quant_type = "unknown";
    for (int64_t i = 0; i < n_tensors; i++) {
        const char * tname = gguf_get_tensor_name(ctx, i);
        // Look for a block weight (e.g. blk.0.attn_q.weight) to get the actual quant
        if (strstr(tname, "blk.") != nullptr && strstr(tname, ".weight") != nullptr) {
            quant_type = ggml_type_name(gguf_get_tensor_type(ctx, i));
            break;
        }
    }
    // Fallback: first weight tensor
    if (quant_type == "unknown") {
        for (int64_t i = 0; i < n_tensors; i++) {
            const char * tname = gguf_get_tensor_name(ctx, i);
            if (strstr(tname, ".weight") != nullptr) {
                quant_type = ggml_type_name(gguf_get_tensor_type(ctx, i));
                break;
            }
        }
    }

    // Calculate total file size from tensor data
    size_t total_tensor_bytes = 0;
    for (int64_t i = 0; i < n_tensors; i++) {
        total_tensor_bytes += gguf_get_tensor_size(ctx, i);
    }

    // Estimate params (total bytes / bytes_per_param based on quant)
    // Rough: for Q8_0 ~1 byte/param, Q4_0 ~0.5 byte/param, F16 ~2 byte/param
    double params_est = 0;
    if (quant_type == "q8_0" || quant_type == "Q8_0") params_est = total_tensor_bytes / 1.0625;
    else if (quant_type == "f16" || quant_type == "F16") params_est = total_tensor_bytes / 2.0;
    else if (quant_type == "f32" || quant_type == "F32") params_est = total_tensor_bytes / 4.0;
    else params_est = total_tensor_bytes / 1.0;  // rough

    std::string params_str;
    if (params_est >= 1e9) {
        char buf[32]; snprintf(buf, sizeof(buf), "%.1fB", params_est / 1e9);
        params_str = buf;
    } else if (params_est >= 1e6) {
        char buf[32]; snprintf(buf, sizeof(buf), "%.0fM", params_est / 1e6);
        params_str = buf;
    } else {
        char buf[32]; snprintf(buf, sizeof(buf), "%.0fK", params_est / 1e3);
        params_str = buf;
    }

    // Get file size
    FILE * f = fopen(filepath.c_str(), "rb");
    size_t file_size = 0;
    if (f) {
        fseek(f, 0, SEEK_END);
        file_size = ftell(f);
        fclose(f);
    }

    // Build layers from tensor names
    // Tensors named like "blk.0.attn_q.weight" → layer 0
    std::map<int, std::vector<int64_t>> layer_tensors;
    std::vector<int64_t> other_tensors;

    for (int64_t i = 0; i < n_tensors; i++) {
        const char * tname = gguf_get_tensor_name(ctx, i);
        // Try to extract layer number from "blk.N." pattern
        const char * blk = strstr(tname, "blk.");
        if (blk) {
            int layer_idx = atoi(blk + 4);
            layer_tensors[layer_idx].push_back(i);
        } else {
            other_tensors.push_back(i);
        }
    }

    // Extract filename from path
    std::string filename = filepath;
    size_t last_sep = filepath.find_last_of("/\\");
    if (last_sep != std::string::npos) filename = filepath.substr(last_sep + 1);

    // Build JSON
    std::ostringstream json;
    json << "{";
    json << "\"filename\":\"" << json_escape(filename) << "\",";
    json << "\"filepath\":\"" << json_escape(filepath) << "\",";
    json << "\"arch\":\"" << json_escape(arch) << "\",";
    json << "\"name\":\"" << json_escape(name) << "\",";
    json << "\"quantType\":\"" << json_escape(quant_type) << "\",";
    json << "\"params\":\"" << json_escape(params_str) << "\",";
    json << "\"contextLength\":" << n_ctx << ",";
    json << "\"embeddingSize\":" << n_embd << ",";
    json << "\"layerCount\":" << n_layer << ",";
    json << "\"headCount\":" << n_head << ",";
    json << "\"headCountKV\":" << n_head_kv << ",";
    json << "\"vocabSize\":" << n_vocab << ",";
    json << "\"fileSize\":" << file_size << ",";
    json << "\"tensorCount\":" << n_tensors << ",";

    // All KV metadata
    json << "\"metadata\":{";
    int64_t n_kv = gguf_get_n_kv(ctx);
    for (int64_t i = 0; i < n_kv; i++) {
        if (i > 0) json << ",";
        const char * key = gguf_get_key(ctx, i);
        json << "\"" << json_escape(key) << "\":" << kv_to_json_value(ctx, i);
    }
    json << "},";

    // Layers
    json << "\"layers\":[";
    bool first_layer = true;
    for (auto & [idx, tensors] : layer_tensors) {
        if (!first_layer) json << ",";
        first_layer = false;

        json << "{\"index\":" << idx << ",\"name\":\"Block " << idx << "\",\"tensors\":[";
        for (size_t t = 0; t < tensors.size(); t++) {
            if (t > 0) json << ",";
            int64_t tid = tensors[t];
            const char * tname = gguf_get_tensor_name(ctx, tid);
            enum ggml_type ttype = gguf_get_tensor_type(ctx, tid);
            size_t tsize = gguf_get_tensor_size(ctx, tid);

            // Get shape from ggml_context if available
            json << "{\"name\":\"" << json_escape(tname) << "\",";
            json << "\"type\":\"" << ggml_type_name(ttype) << "\",";
            json << "\"size\":" << tsize << ",";
            json << "\"shape\":[";
            if (ggml_ctx) {
                struct ggml_tensor * tensor = ggml_get_tensor(ggml_ctx, tname);
                if (tensor) {
                    int n_dims = ggml_n_dims(tensor);
                    for (int d = 0; d < n_dims; d++) {
                        if (d > 0) json << ",";
                        json << tensor->ne[d];
                    }
                }
            }
            json << "]}";
        }
        json << "]}";
    }
    json << "],";

    // Flat tensor list (for tensor table view)
    json << "\"tensors\":[";
    for (int64_t i = 0; i < n_tensors; i++) {
        if (i > 0) json << ",";
        const char * tname = gguf_get_tensor_name(ctx, i);
        enum ggml_type ttype = gguf_get_tensor_type(ctx, i);
        size_t tsize = gguf_get_tensor_size(ctx, i);

        json << "{\"name\":\"" << json_escape(tname) << "\",";
        json << "\"type\":\"" << ggml_type_name(ttype) << "\",";
        json << "\"size\":" << tsize << ",";
        json << "\"shape\":[";
        if (ggml_ctx) {
            struct ggml_tensor * tensor = ggml_get_tensor(ggml_ctx, tname);
            if (tensor) {
                int n_dims = ggml_n_dims(tensor);
                for (int d = 0; d < n_dims; d++) {
                    if (d > 0) json << ",";
                    json << tensor->ne[d];
                }
            }
        }
        json << "]}";
    }
    json << "]";

    json << "}";

    if (ggml_ctx) ggml_free(ggml_ctx);
    gguf_free(ctx);
    return json.str();
}

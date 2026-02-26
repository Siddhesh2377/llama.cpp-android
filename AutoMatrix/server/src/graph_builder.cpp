#include "graph_builder.h"
#include "gguf.h"
#include "ggml.h"

#include <cstring>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <algorithm>

static std::string je(const std::string & s) {
    std::string out;
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else out += c;
    }
    return out;
}

struct GraphNode {
    std::string id;
    std::string label;
    std::string type;      // embed, attn, ffn, norm, head, proj
    std::string sublabel;
    int layer;             // -1 for non-layer nodes
    int x, y;              // layout position
    int width, height;
};

struct GraphEdge {
    std::string from;
    std::string to;
};

// Categorize a tensor name into a node type
static std::string categorize_tensor(const char * name) {
    // Block-level tensors (contain "blk.")
    if (strstr(name, "blk.")) {
        if (strstr(name, "attn_q") || strstr(name, "attn_k") || strstr(name, "attn_v") ||
            strstr(name, "attn_qkv")) return "attn";
        if (strstr(name, "attn_output") || strstr(name, "attn_o.")) return "attn_proj";
        if (strstr(name, "attn_norm") || strstr(name, "attn_ln")) return "attn_norm";
        if (strstr(name, "ffn_gate") || strstr(name, "ffn_up")) return "ffn_gate";
        if (strstr(name, "ffn_down")) return "ffn_down";
        if (strstr(name, "ffn_norm")) return "ffn_norm";
        return "other";
    }
    // Top-level tensors
    if (strstr(name, "output_norm") || strstr(name, "final_norm")) return "final_norm";
    if (strstr(name, "output") && strstr(name, ".weight")) return "head";
    if (strstr(name, "lm_head")) return "head";
    if (strstr(name, "token_embd") || strstr(name, "embed")) return "embed";
    return "other";
}

std::string build_model_graph(const std::string & filepath) {
    struct gguf_init_params params = { true, nullptr };
    struct gguf_context * ctx = gguf_init_from_file(filepath.c_str(), params);
    if (!ctx) return "";

    int64_t n_tensors = gguf_get_n_tensors(ctx);

    // Find architecture info
    int64_t arch_id = gguf_find_key(ctx, "general.architecture");
    std::string arch = "unknown";
    if (arch_id >= 0 && gguf_get_kv_type(ctx, arch_id) == GGUF_TYPE_STRING) {
        arch = gguf_get_val_str(ctx, arch_id);
    }

    // Get block count
    uint32_t n_layers = 0;
    int64_t bc_id = gguf_find_key(ctx, (arch + ".block_count").c_str());
    if (bc_id >= 0) {
        auto t = gguf_get_kv_type(ctx, bc_id);
        if (t == GGUF_TYPE_UINT32) n_layers = gguf_get_val_u32(ctx, bc_id);
        else if (t == GGUF_TYPE_INT32) n_layers = (uint32_t)gguf_get_val_i32(ctx, bc_id);
    }

    // Group tensors by layer and type
    struct TensorGroup {
        std::string type;
        int layer;
        std::vector<std::string> tensors;
        size_t total_bytes;
    };

    std::map<std::string, TensorGroup> groups;

    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(ctx, i);
        std::string cat = categorize_tensor(name);
        size_t tsize = gguf_get_tensor_size(ctx, i);

        // Extract layer number
        int layer = -1;
        const char * blk = strstr(name, "blk.");
        if (blk) layer = atoi(blk + 4);

        std::string group_key;
        if (layer >= 0) {
            group_key = "L" + std::to_string(layer) + "_" + cat;
        } else {
            group_key = cat;
        }

        auto & g = groups[group_key];
        g.type = cat;
        g.layer = layer;
        g.tensors.push_back(name);
        g.total_bytes += tsize;
    }

    // Build nodes and edges
    std::vector<GraphNode> nodes;
    std::vector<GraphEdge> edges;

    // Layout parameters
    const int col_w = 180;
    const int row_h = 52;
    const int node_w = 160;
    const int node_h = 40;
    const int layer_group_h = row_h * 6;  // 6 sub-nodes per layer
    const int start_y = 20;

    int next_id = 0;
    auto make_id = [&]() { return "n" + std::to_string(next_id++); };

    // Format bytes
    auto fmt_bytes = [](size_t b) -> std::string {
        char buf[32];
        if (b >= 1024*1024) snprintf(buf, sizeof(buf), "%.1fMB", b / (1024.0*1024.0));
        else if (b >= 1024) snprintf(buf, sizeof(buf), "%.1fKB", b / 1024.0);
        else snprintf(buf, sizeof(buf), "%zuB", b);
        return buf;
    };

    // Embedding node
    std::string embed_id = make_id();
    int center_x = 300;
    int cur_y = start_y;

    if (groups.count("embed")) {
        nodes.push_back({embed_id, "Token Embedding", "embed",
            fmt_bytes(groups["embed"].total_bytes), -1,
            center_x - node_w/2, cur_y, node_w, node_h});
        cur_y += row_h + 10;
    }

    // Per-layer nodes
    std::string prev_layer_out = embed_id;

    // Only show first 3 layers + "..." + last layer if many layers
    std::vector<int> layers_to_show;
    if (n_layers <= 6) {
        for (uint32_t i = 0; i < n_layers; i++) layers_to_show.push_back(i);
    } else {
        layers_to_show = {0, 1, 2, -1, (int)n_layers - 2, (int)n_layers - 1};
    }

    for (int li : layers_to_show) {
        if (li == -1) {
            // Ellipsis node
            std::string eid = make_id();
            nodes.push_back({eid, "...", "ellipsis",
                std::to_string(n_layers - 5) + " more layers", -1,
                center_x - node_w/2, cur_y, node_w, node_h});
            edges.push_back({prev_layer_out, eid});
            prev_layer_out = eid;
            cur_y += row_h + 4;
            continue;
        }

        std::string prefix = "L" + std::to_string(li) + "_";

        // Attention Norm
        std::string attn_norm_id = make_id();
        std::string attn_norm_sub;
        if (groups.count(prefix + "attn_norm"))
            attn_norm_sub = fmt_bytes(groups[prefix + "attn_norm"].total_bytes);
        nodes.push_back({attn_norm_id, "Attn Norm L" + std::to_string(li), "norm",
            attn_norm_sub, li, center_x - node_w/2, cur_y, node_w, node_h});
        edges.push_back({prev_layer_out, attn_norm_id});
        cur_y += row_h;

        // QKV Attention
        std::string attn_id = make_id();
        std::string attn_sub;
        if (groups.count(prefix + "attn"))
            attn_sub = fmt_bytes(groups[prefix + "attn"].total_bytes);
        nodes.push_back({attn_id, "Attention L" + std::to_string(li), "attn",
            attn_sub, li, center_x - node_w/2, cur_y, node_w, node_h});
        edges.push_back({attn_norm_id, attn_id});
        cur_y += row_h;

        // Attention Output Projection
        std::string attn_proj_id = make_id();
        if (groups.count(prefix + "attn_proj")) {
            nodes.push_back({attn_proj_id, "Attn Proj L" + std::to_string(li), "proj",
                fmt_bytes(groups[prefix + "attn_proj"].total_bytes), li,
                center_x - node_w/2, cur_y, node_w, node_h});
            edges.push_back({attn_id, attn_proj_id});
            cur_y += row_h;
        } else {
            attn_proj_id = attn_id;
        }

        // FFN Norm
        std::string ffn_norm_id = make_id();
        if (groups.count(prefix + "ffn_norm")) {
            nodes.push_back({ffn_norm_id, "FFN Norm L" + std::to_string(li), "norm",
                fmt_bytes(groups[prefix + "ffn_norm"].total_bytes), li,
                center_x - node_w/2, cur_y, node_w, node_h});
            edges.push_back({attn_proj_id, ffn_norm_id});
            cur_y += row_h;
        } else {
            ffn_norm_id = attn_proj_id;
        }

        // FFN Gate/Up
        std::string ffn_gate_id = make_id();
        if (groups.count(prefix + "ffn_gate")) {
            nodes.push_back({ffn_gate_id, "FFN Gate+Up L" + std::to_string(li), "ffn",
                fmt_bytes(groups[prefix + "ffn_gate"].total_bytes), li,
                center_x - node_w/2, cur_y, node_w, node_h});
            edges.push_back({ffn_norm_id, ffn_gate_id});
            cur_y += row_h;
        } else {
            ffn_gate_id = ffn_norm_id;
        }

        // FFN Down
        std::string ffn_down_id = make_id();
        if (groups.count(prefix + "ffn_down")) {
            nodes.push_back({ffn_down_id, "FFN Down L" + std::to_string(li), "ffn",
                fmt_bytes(groups[prefix + "ffn_down"].total_bytes), li,
                center_x - node_w/2, cur_y, node_w, node_h});
            edges.push_back({ffn_gate_id, ffn_down_id});
            cur_y += row_h;
        } else {
            ffn_down_id = ffn_gate_id;
        }

        prev_layer_out = ffn_down_id;
        cur_y += 8;  // layer gap
    }

    // Final norm
    std::string final_norm_id = make_id();
    if (groups.count("final_norm")) {
        nodes.push_back({final_norm_id, "Output Norm", "norm",
            fmt_bytes(groups["final_norm"].total_bytes), -1,
            center_x - node_w/2, cur_y, node_w, node_h});
        edges.push_back({prev_layer_out, final_norm_id});
        cur_y += row_h;
        prev_layer_out = final_norm_id;
    }

    // LM Head
    if (groups.count("head")) {
        std::string head_id = make_id();
        nodes.push_back({head_id, "LM Head", "head",
            fmt_bytes(groups["head"].total_bytes), -1,
            center_x - node_w/2, cur_y, node_w, node_h});
        edges.push_back({prev_layer_out, head_id});
    }

    // Build JSON
    std::ostringstream json;
    json << "{\"nodes\":[";
    for (size_t i = 0; i < nodes.size(); i++) {
        if (i > 0) json << ",";
        auto & n = nodes[i];
        json << "{\"id\":\"" << je(n.id) << "\",";
        json << "\"label\":\"" << je(n.label) << "\",";
        json << "\"type\":\"" << je(n.type) << "\",";
        json << "\"sublabel\":\"" << je(n.sublabel) << "\",";
        json << "\"layer\":" << n.layer << ",";
        json << "\"x\":" << n.x << ",\"y\":" << n.y << ",";
        json << "\"width\":" << n.width << ",\"height\":" << n.height << "}";
    }
    json << "],\"edges\":[";
    for (size_t i = 0; i < edges.size(); i++) {
        if (i > 0) json << ",";
        json << "{\"from\":\"" << je(edges[i].from) << "\",\"to\":\"" << je(edges[i].to) << "\"}";
    }
    json << "],\"layerCount\":" << n_layers << ",\"arch\":\"" << je(arch) << "\"}";

    gguf_free(ctx);
    return json.str();
}

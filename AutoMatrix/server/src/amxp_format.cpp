#include "amxp_format.h"

#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <sys/stat.h>

static const char AMXP_MAGIC[4] = {'A', 'M', 'X', 'P'};
static const uint16_t AMXP_VERSION = 1;
static const size_t AMXP_HEADER_SIZE = 16;

// Quick JSON string field extractor (no full parser needed)
static std::string extract_json_str(const std::string & json, const char * key) {
    // Search for "key":"value"
    std::string needle = std::string("\"") + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return "";

    // Skip past key and colon
    pos += needle.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == ':')) pos++;
    if (pos >= json.size() || json[pos] != '"') return "";

    pos++; // skip opening quote
    std::string result;
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            result += json[++pos];
        } else {
            result += json[pos];
        }
        pos++;
    }
    return result;
}

std::string AmxpPlugin::id() const {
    return extract_json_str(json_body, "id");
}

std::string AmxpPlugin::name() const {
    return extract_json_str(json_body, "name");
}

AmxpPlugin amxp_read(const std::string & filepath) {
    AmxpPlugin p = {};
    p.filepath = filepath;

    FILE * f = fopen(filepath.c_str(), "rb");
    if (!f) return p;

    // Read header
    uint8_t header[AMXP_HEADER_SIZE];
    if (fread(header, 1, AMXP_HEADER_SIZE, f) != AMXP_HEADER_SIZE) {
        fclose(f);
        return p;
    }

    // Verify magic
    if (memcmp(header, AMXP_MAGIC, 4) != 0) {
        fclose(f);
        return p;
    }

    // Parse header (little-endian)
    p.version   = (uint16_t)(header[4] | (header[5] << 8));
    p.type      = (AmxpType)(uint16_t)(header[6] | (header[7] << 8));
    p.flags     = (uint32_t)(header[8] | (header[9] << 8) | (header[10] << 16) | (header[11] << 24));
    uint32_t body_len = (uint32_t)(header[12] | (header[13] << 8) | (header[14] << 16) | (header[15] << 24));

    // Sanity check
    if (body_len > 10 * 1024 * 1024) { // 10MB max
        fclose(f);
        return p;
    }

    // Read body
    p.json_body.resize(body_len);
    if (fread(&p.json_body[0], 1, body_len, f) != body_len) {
        p.json_body.clear();
        fclose(f);
        return p;
    }

    fclose(f);
    return p;
}

bool amxp_write(const std::string & filepath, AmxpType type, const std::string & json_body, uint32_t flags) {
    FILE * f = fopen(filepath.c_str(), "wb");
    if (!f) return false;

    uint32_t body_len = (uint32_t)json_body.size();

    // Write header
    uint8_t header[AMXP_HEADER_SIZE];
    memcpy(header, AMXP_MAGIC, 4);
    header[4]  = (uint8_t)(AMXP_VERSION & 0xFF);
    header[5]  = (uint8_t)((AMXP_VERSION >> 8) & 0xFF);
    header[6]  = (uint8_t)((uint16_t)type & 0xFF);
    header[7]  = (uint8_t)(((uint16_t)type >> 8) & 0xFF);
    header[8]  = (uint8_t)(flags & 0xFF);
    header[9]  = (uint8_t)((flags >> 8) & 0xFF);
    header[10] = (uint8_t)((flags >> 16) & 0xFF);
    header[11] = (uint8_t)((flags >> 24) & 0xFF);
    header[12] = (uint8_t)(body_len & 0xFF);
    header[13] = (uint8_t)((body_len >> 8) & 0xFF);
    header[14] = (uint8_t)((body_len >> 16) & 0xFF);
    header[15] = (uint8_t)((body_len >> 24) & 0xFF);

    if (fwrite(header, 1, AMXP_HEADER_SIZE, f) != AMXP_HEADER_SIZE) {
        fclose(f);
        return false;
    }

    // Write body
    if (fwrite(json_body.data(), 1, body_len, f) != body_len) {
        fclose(f);
        return false;
    }

    fclose(f);
    return true;
}

std::vector<AmxpPlugin> amxp_scan_dir(const std::string & dirpath, int type_filter) {
    std::vector<AmxpPlugin> result;

    // Scan subdirectories: architectures/, backends/, quant_types/, sampling/
    const char * subdirs[] = {"architectures", "backends", "quant_types", "sampling"};

    for (const char * sub : subdirs) {
        std::string path = dirpath + "/" + sub;
        DIR * d = opendir(path.c_str());
        if (!d) continue;

        struct dirent * entry;
        while ((entry = readdir(d)) != nullptr) {
            std::string name = entry->d_name;
            if (name.size() < 6 || name.substr(name.size() - 5) != ".amxp") continue;

            std::string fpath = path + "/" + name;
            AmxpPlugin p = amxp_read(fpath);
            if (p.json_body.empty()) continue;

            if (type_filter >= 0 && (int)p.type != type_filter) continue;
            result.push_back(std::move(p));
        }
        closedir(d);
    }

    return result;
}

const char * amxp_type_str(AmxpType type) {
    switch (type) {
        case AMXP_ARCH:     return "architecture";
        case AMXP_BACKEND:  return "backend";
        case AMXP_QUANT:    return "quant";
        case AMXP_SAMPLING: return "sampling";
        default:            return "unknown";
    }
}

const char * amxp_type_dir(AmxpType type) {
    switch (type) {
        case AMXP_ARCH:     return "architectures";
        case AMXP_BACKEND:  return "backends";
        case AMXP_QUANT:    return "quant_types";
        case AMXP_SAMPLING: return "sampling";
        default:            return "";
    }
}

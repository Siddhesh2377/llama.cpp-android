#pragma once

#include <string>
#include <vector>
#include <cstdint>

// AMXP — AutoMatrix Plugin format
// 16-byte binary header + JSON body
//
// Header layout:
//   [0..3]   Magic: "AMXP"
//   [4..5]   Version: u16 LE (currently 1)
//   [6..7]   Type: u16 LE (0=arch, 1=backend, 2=quant, 3=sampling)
//   [8..11]  Flags: u32 LE (reserved, 0)
//   [12..15] Body length: u32 LE (JSON byte count)
//   [16..]   JSON body (UTF-8, no null terminator)

enum AmxpType : uint16_t {
    AMXP_ARCH     = 0,
    AMXP_BACKEND  = 1,
    AMXP_QUANT    = 2,
    AMXP_SAMPLING = 3,
};

struct AmxpPlugin {
    uint16_t    version;
    AmxpType    type;
    uint32_t    flags;
    std::string json_body;
    std::string filepath;   // set by reader

    // Convenience: extract "id" from JSON body (fast string scan, no full parse)
    std::string id() const;
    std::string name() const;
};

// Read a single .amxp file. Returns empty plugin (json_body.empty()) on failure.
AmxpPlugin amxp_read(const std::string & filepath);

// Write a .amxp file. Returns true on success.
bool amxp_write(const std::string & filepath, AmxpType type, const std::string & json_body, uint32_t flags = 0);

// Scan a directory for all .amxp files, optionally filtering by type (-1 = all).
std::vector<AmxpPlugin> amxp_scan_dir(const std::string & dirpath, int type_filter = -1);

// Type enum to string
const char * amxp_type_str(AmxpType type);

// Type string to subdirectory name
const char * amxp_type_dir(AmxpType type);

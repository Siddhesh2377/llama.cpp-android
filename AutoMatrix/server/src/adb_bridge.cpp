#include "adb_bridge.h"

#include <cstdio>
#include <cstring>
#include <sstream>
#include <vector>

static std::string exec_cmd(const char * cmd) {
    FILE * pipe = popen(cmd, "r");
    if (!pipe) return "";
    char buf[256];
    std::string result;
    while (fgets(buf, sizeof(buf), pipe)) {
        result += buf;
    }
    pclose(pipe);
    return result;
}

static std::string json_escape(const std::string & s) {
    std::string out;
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n' || c == '\r') continue;
        else out += c;
    }
    return out;
}

static std::string adb_getprop(const std::string & serial, const char * prop) {
    std::string cmd = "adb -s " + serial + " shell getprop " + prop + " 2>/dev/null";
    std::string val = exec_cmd(cmd.c_str());
    // Trim whitespace/newlines
    while (!val.empty() && (val.back() == '\n' || val.back() == '\r' || val.back() == ' '))
        val.pop_back();
    return val;
}

std::string get_adb_devices() {
    std::string output = exec_cmd("adb devices -l 2>/dev/null");
    if (output.empty()) {
        return "{\"devices\":[]}";
    }

    std::ostringstream json;
    json << "{\"devices\":[";

    std::istringstream stream(output);
    std::string line;
    bool first = true;

    while (std::getline(stream, line)) {
        // Skip header and empty lines
        if (line.find("List of devices") != std::string::npos) continue;
        if (line.empty() || line[0] == '*') continue;

        // Parse: "SERIAL\tSTATUS..." or "SERIAL STATUS..." (adb-tls uses spaces)
        size_t sep = line.find('\t');
        if (sep == std::string::npos) {
            // Try space-separated: find "device" or "unauthorized" or "offline" keyword
            for (const char * kw : {"device ", "unauthorized ", "offline "}) {
                sep = line.find(kw);
                if (sep != std::string::npos) break;
            }
        }
        if (sep == std::string::npos) continue;

        std::string serial = line.substr(0, sep);
        // Trim trailing whitespace from serial
        while (!serial.empty() && serial.back() == ' ') serial.pop_back();
        std::string rest = line.substr(sep);
        // Trim leading whitespace/tab
        while (!rest.empty() && (rest[0] == ' ' || rest[0] == '\t')) rest.erase(rest.begin());

        // Extract status (first word after tab)
        std::string status = "offline";
        size_t sp = rest.find(' ');
        if (sp != std::string::npos) {
            status = rest.substr(0, sp);
        } else {
            status = rest;
        }
        // Trim
        while (!status.empty() && (status.back() == '\n' || status.back() == '\r' || status.back() == ' '))
            status.pop_back();

        // Map status
        std::string status_mapped;
        if (status == "device")       status_mapped = "online";
        else if (status == "unauthorized") status_mapped = "unauthorized";
        else                          status_mapped = "offline";

        // Extract model from the line (model:XXX)
        std::string model_name;
        size_t model_pos = rest.find("model:");
        if (model_pos != std::string::npos) {
            size_t start = model_pos + 6;
            size_t end = rest.find(' ', start);
            model_name = rest.substr(start, end - start);
        }

        // Get additional props for online devices
        std::string chipset, ram, abi;
        if (status_mapped == "online") {
            chipset = adb_getprop(serial, "ro.hardware.chipname");
            if (chipset.empty()) chipset = adb_getprop(serial, "ro.board.platform");
            ram = adb_getprop(serial, "ro.totalram");  // May not exist
            abi = adb_getprop(serial, "ro.product.cpu.abilist");
            if (model_name.empty()) {
                model_name = adb_getprop(serial, "ro.product.model");
            }
        }

        if (!first) json << ",";
        first = false;

        json << "{";
        json << "\"serial\":\"" << json_escape(serial) << "\",";
        json << "\"model\":\"" << json_escape(model_name) << "\",";
        json << "\"chipset\":\"" << json_escape(chipset) << "\",";
        json << "\"ram\":\"" << json_escape(ram) << "\",";
        json << "\"status\":\"" << status_mapped << "\",";
        json << "\"abiList\":\"" << json_escape(abi) << "\"";
        json << "}";
    }

    json << "]}";
    return json.str();
}

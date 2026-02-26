#include "httplib.h"
#include "gguf_parser.h"
#include "adb_bridge.h"
#include "graph_builder.h"
#include "amxp_format.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>

#ifdef __linux__
#include <sys/utsname.h>
#include <unistd.h>
#include <sys/sysinfo.h>
#endif

// Current loaded model state (cached)
static std::string g_model_json;
static std::string g_model_filepath;

static std::string json_escape_str(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8]; snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += (char)c;
                }
        }
    }
    return out;
}

struct FileEntry {
    std::string path;
    std::string name;
    size_t size;
};

static void scan_dir_recursive(const std::string & dir, std::vector<FileEntry> & results, int depth = 0) {
    if (depth > 5) return;  // limit recursion
    DIR * d = opendir(dir.c_str());
    if (!d) return;

    struct dirent * ent;
    while ((ent = readdir(d)) != nullptr) {
        if (ent->d_name[0] == '.') continue;  // skip hidden
        std::string full = dir + "/" + ent->d_name;

        struct stat st;
        if (stat(full.c_str(), &st) != 0) continue;

        if (S_ISDIR(st.st_mode)) {
            scan_dir_recursive(full, results, depth + 1);
        } else if (S_ISREG(st.st_mode)) {
            std::string name = ent->d_name;
            // Check for .gguf extension
            if (name.size() > 5 && name.substr(name.size() - 5) == ".gguf") {
                results.push_back({full, name, (size_t)st.st_size});
            }
        }

        if (results.size() >= 200) break;  // safety limit
    }
    closedir(d);
}

static std::string scan_for_models(const std::string & directory) {
    std::vector<FileEntry> files;
    scan_dir_recursive(directory, files);

    std::ostringstream json;
    json << "{\"directory\":\"" << json_escape_str(directory) << "\",\"files\":[";
    for (size_t i = 0; i < files.size(); i++) {
        if (i > 0) json << ",";
        json << "{\"path\":\"" << json_escape_str(files[i].path) << "\",";
        json << "\"name\":\"" << json_escape_str(files[i].name) << "\",";
        json << "\"size\":" << files[i].size << "}";
    }
    json << "]}";
    return json.str();
}

static std::string get_system_info() {
    std::ostringstream json;
    json << "{";

    char hostname[256] = {0};
    gethostname(hostname, sizeof(hostname));
    json << "\"hostname\":\"" << hostname << "\",";

#ifdef __linux__
    struct utsname uts;
    uname(&uts);
    json << "\"platform\":\"" << uts.sysname << " " << uts.release << "\",";

    // CPU model
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string cpu_model = "unknown";
    if (cpuinfo.is_open()) {
        std::string line;
        while (std::getline(cpuinfo, line)) {
            if (line.find("model name") != std::string::npos) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    cpu_model = line.substr(colon + 2);
                }
                break;
            }
        }
    }
    json << "\"cpuModel\":\"" << cpu_model << "\",";

    // CPU cores
    int cores = sysconf(_SC_NPROCESSORS_ONLN);
    json << "\"cpuCores\":" << cores << ",";

    // Memory
    struct sysinfo si;
    sysinfo(&si);
    json << "\"memoryTotal\":" << (si.totalram * si.mem_unit) << ",";
    json << "\"memoryFree\":" << (si.freeram * si.mem_unit);
#else
    json << "\"platform\":\"unknown\",";
    json << "\"cpuModel\":\"unknown\",";
    json << "\"cpuCores\":1,";
    json << "\"memoryTotal\":0,";
    json << "\"memoryFree\":0";
#endif

    json << "}";
    return json.str();
}

// Extract a string field from JSON body (minimal parser)
static std::string json_get_string(const std::string & body, const char * key) {
    std::string search = std::string("\"") + key + "\"";
    size_t pos = body.find(search);
    if (pos == std::string::npos) return "";

    pos = body.find(':', pos);
    if (pos == std::string::npos) return "";

    pos = body.find('"', pos + 1);
    if (pos == std::string::npos) return "";

    size_t end = body.find('"', pos + 1);
    if (end == std::string::npos) return "";

    return body.substr(pos + 1, end - pos - 1);
}

int main(int argc, char ** argv) {
    int port = 8080;

    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
        }
    }

    httplib::Server svr;

    // CORS middleware
    svr.set_pre_routing_handler([](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        if (req.method == "OPTIONS") {
            res.status = 204;
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    // GET /api/health
    svr.Get("/api/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content("{\"status\":\"ok\",\"version\":\"0.3.0\"}", "application/json");
    });

    // GET /api/system
    svr.Get("/api/system", [](const httplib::Request &, httplib::Response & res) {
        std::string info = get_system_info();
        res.set_content(info, "application/json");
    });

    // POST /api/model/load
    svr.Post("/api/model/load", [](const httplib::Request & req, httplib::Response & res) {
        std::string filepath = json_get_string(req.body, "filepath");
        if (filepath.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"missing filepath\"}", "application/json");
            return;
        }

        printf("[amx-server] Loading model: %s\n", filepath.c_str());
        std::string json = parse_gguf(filepath);
        if (json.empty()) {
            res.status = 500;
            res.set_content("{\"error\":\"failed to parse GGUF file\"}", "application/json");
            return;
        }

        g_model_json = json;
        g_model_filepath = filepath;
        printf("[amx-server] Model loaded successfully (%zu bytes JSON)\n", json.size());
        res.set_content(json, "application/json");
    });

    // GET /api/model/info
    svr.Get("/api/model/info", [](const httplib::Request &, httplib::Response & res) {
        if (g_model_json.empty()) {
            res.status = 404;
            res.set_content("{\"error\":\"no model loaded\"}", "application/json");
            return;
        }
        res.set_content(g_model_json, "application/json");
    });

    // GET /api/devices
    svr.Get("/api/devices", [](const httplib::Request &, httplib::Response & res) {
        std::string devices = get_adb_devices();
        res.set_content(devices, "application/json");
    });

    // GET /api/model/graph — compute graph for loaded model
    svr.Get("/api/model/graph", [](const httplib::Request &, httplib::Response & res) {
        if (g_model_filepath.empty()) {
            res.status = 404;
            res.set_content("{\"error\":\"no model loaded\"}", "application/json");
            return;
        }
        std::string graph = build_model_graph(g_model_filepath);
        if (graph.empty()) {
            res.status = 500;
            res.set_content("{\"error\":\"failed to build graph\"}", "application/json");
            return;
        }
        res.set_content(graph, "application/json");
    });

    // POST /api/models/scan — scan directory for GGUF files
    svr.Post("/api/models/scan", [](const httplib::Request & req, httplib::Response & res) {
        std::string directory = json_get_string(req.body, "directory");
        if (directory.empty()) {
            // Default: scan common locations
            const char * home = getenv("HOME");
            if (home) directory = std::string(home);
            else directory = "/tmp";
        }
        printf("[amx-server] Scanning for models in: %s\n", directory.c_str());
        std::string json = scan_for_models(directory);
        res.set_content(json, "application/json");
    });

    // POST /api/deploy/push — push vlm-test binary to device
    svr.Post("/api/deploy/push", [](const httplib::Request & req, httplib::Response & res) {
        std::string serial = json_get_string(req.body, "serial");
        std::string binary_dir = json_get_string(req.body, "binaryDir");
        if (binary_dir.empty()) binary_dir = "build-android/bin";

        std::string serial_arg = serial.empty() ? "" : "-s " + serial + " ";
        std::string cmd = "adb " + serial_arg + "push " + binary_dir + "/vlm-test " +
                          binary_dir + "/lib*.so /data/local/tmp/ 2>&1";
        printf("[amx-server] Deploy: %s\n", cmd.c_str());

        FILE * pipe = popen(cmd.c_str(), "r");
        if (!pipe) {
            res.status = 500;
            res.set_content("{\"error\":\"failed to run adb push\"}", "application/json");
            return;
        }
        char buf[512];
        std::string output;
        while (fgets(buf, sizeof(buf), pipe)) output += buf;
        int rc = pclose(pipe);

        std::ostringstream json;
        json << "{\"success\":" << (rc == 0 ? "true" : "false");
        json << ",\"output\":\"" << json_escape_str(output) << "\"}";
        res.set_content(json.str(), "application/json");
    });

    // POST /api/inference/run — run inference on device, stream output via SSE
    svr.Post("/api/inference/run", [](const httplib::Request & req, httplib::Response & res) {
        std::string serial = json_get_string(req.body, "serial");
        std::string model_path = json_get_string(req.body, "modelPath");
        std::string mmproj_path = json_get_string(req.body, "mmprojPath");
        std::string image_path = json_get_string(req.body, "imagePath");
        std::string prompt = json_get_string(req.body, "prompt");
        std::string threads = json_get_string(req.body, "threads");
        std::string quant = json_get_string(req.body, "quant");
        std::string max_tokens = json_get_string(req.body, "maxTokens");

        if (model_path.empty()) model_path = "/sdcard/Download/SmolVLM-500M-Instruct-q8_0.gguf";
        if (threads.empty()) threads = "4";
        if (max_tokens.empty()) max_tokens = "64";

        std::string serial_arg = serial.empty() ? "" : "-s " + serial + " ";

        // Build vlm-test command
        std::string device_cmd = "cd /data/local/tmp && LD_LIBRARY_PATH=. ./vlm-test"
            " --model " + model_path +
            " --threads " + threads +
            " --max-tokens " + max_tokens;

        if (!mmproj_path.empty()) device_cmd += " --mmproj " + mmproj_path;
        if (!image_path.empty()) device_cmd += " --image " + image_path;
        if (quant == "q5") device_cmd += " --q5";
        else if (quant == "q4") device_cmd += " --q4";
        if (!prompt.empty()) device_cmd += " --prompt \"" + prompt + "\"";

        std::string cmd = "adb " + serial_arg + "shell \"" + device_cmd + "\" 2>&1";
        printf("[amx-server] Inference: %s\n", cmd.c_str());

        FILE * pipe = popen(cmd.c_str(), "r");
        if (!pipe) {
            res.status = 500;
            res.set_content("{\"error\":\"failed to start inference\"}", "application/json");
            return;
        }

        // Stream output line by line
        res.set_header("Content-Type", "text/event-stream");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");

        // Collect all output and send as JSON
        char buf[512];
        std::string all_output;
        while (fgets(buf, sizeof(buf), pipe)) {
            all_output += buf;
        }
        int rc = pclose(pipe);

        std::ostringstream json;
        json << "{\"success\":" << (rc == 0 ? "true" : "false");
        json << ",\"output\":\"" << json_escape_str(all_output) << "\"}";
        res.set_content(json.str(), "application/json");
    });

    // Resolve plugins directory (relative to binary or explicit)
    std::string plugins_dir;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--plugins") == 0 && i + 1 < argc) {
            plugins_dir = argv[++i];
        }
    }
    if (plugins_dir.empty()) {
        // Try common locations relative to binary
        const char * candidates[] = {"plugins", "../plugins", "../../plugins",
            "../AutoMatrix/plugins", "../../AutoMatrix/plugins"};
        for (const char * c : candidates) {
            DIR * d = opendir(c);
            if (d) { closedir(d); plugins_dir = c; break; }
        }
    }

    // GET /api/plugins — list all plugins
    svr.Get("/api/plugins", [&plugins_dir](const httplib::Request &, httplib::Response & res) {
        if (plugins_dir.empty()) {
            res.set_content("{\"error\":\"plugins directory not found\"}", "application/json");
            return;
        }
        auto plugins = amxp_scan_dir(plugins_dir);
        std::ostringstream json;
        json << "{\"pluginDir\":\"" << json_escape_str(plugins_dir) << "\",\"plugins\":[";
        for (size_t i = 0; i < plugins.size(); i++) {
            if (i > 0) json << ",";
            json << "{\"id\":\"" << json_escape_str(plugins[i].id()) << "\",";
            json << "\"name\":\"" << json_escape_str(plugins[i].name()) << "\",";
            json << "\"type\":\"" << amxp_type_str(plugins[i].type) << "\",";
            json << "\"version\":" << plugins[i].version << ",";
            json << "\"filepath\":\"" << json_escape_str(plugins[i].filepath) << "\",";
            json << "\"body\":" << plugins[i].json_body << "}";
        }
        json << "]}";
        res.set_content(json.str(), "application/json");
    });

    // GET /api/plugins/:type — list plugins by type
    svr.Get(R"(/api/plugins/(\w+))", [&plugins_dir](const httplib::Request & req, httplib::Response & res) {
        if (plugins_dir.empty()) {
            res.set_content("{\"error\":\"plugins directory not found\"}", "application/json");
            return;
        }
        std::string type_str = req.matches[1];
        int type_filter = -1;
        if (type_str == "architectures" || type_str == "arch") type_filter = AMXP_ARCH;
        else if (type_str == "backends" || type_str == "backend") type_filter = AMXP_BACKEND;
        else if (type_str == "quant_types" || type_str == "quant") type_filter = AMXP_QUANT;
        else if (type_str == "sampling") type_filter = AMXP_SAMPLING;

        auto plugins = amxp_scan_dir(plugins_dir, type_filter);
        std::ostringstream json;
        json << "[";
        for (size_t i = 0; i < plugins.size(); i++) {
            if (i > 0) json << ",";
            json << plugins[i].json_body;
        }
        json << "]";
        res.set_content(json.str(), "application/json");
    });

    // POST /api/plugins/save — save a plugin (create or update)
    svr.Post("/api/plugins/save", [&plugins_dir](const httplib::Request & req, httplib::Response & res) {
        if (plugins_dir.empty()) {
            res.status = 500;
            res.set_content("{\"error\":\"plugins directory not found\"}", "application/json");
            return;
        }
        // Expect: {"type":"arch|backend|quant|sampling", "id":"name", "body":{...}}
        std::string type_str = json_get_string(req.body, "type");
        std::string id = json_get_string(req.body, "id");
        if (type_str.empty() || id.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"missing type or id\"}", "application/json");
            return;
        }

        AmxpType atype;
        if (type_str == "arch" || type_str == "architecture") atype = AMXP_ARCH;
        else if (type_str == "backend") atype = AMXP_BACKEND;
        else if (type_str == "quant") atype = AMXP_QUANT;
        else if (type_str == "sampling") atype = AMXP_SAMPLING;
        else {
            res.status = 400;
            res.set_content("{\"error\":\"invalid type\"}", "application/json");
            return;
        }

        // Extract the "body" JSON object from the request
        size_t body_pos = req.body.find("\"body\"");
        if (body_pos == std::string::npos) {
            res.status = 400;
            res.set_content("{\"error\":\"missing body\"}", "application/json");
            return;
        }
        // Find the opening { after "body":
        size_t brace = req.body.find('{', body_pos + 6);
        if (brace == std::string::npos) { res.status = 400; return; }

        // Match braces to find end of body object
        int depth = 0;
        size_t end = brace;
        for (; end < req.body.size(); end++) {
            if (req.body[end] == '{') depth++;
            else if (req.body[end] == '}') { depth--; if (depth == 0) { end++; break; } }
        }
        std::string json_body = req.body.substr(brace, end - brace);

        std::string filepath = plugins_dir + "/" + amxp_type_dir(atype) + "/" + id + ".amxp";
        bool ok = amxp_write(filepath, atype, json_body);
        if (ok) {
            printf("[amx-server] Saved plugin: %s\n", filepath.c_str());
            res.set_content("{\"success\":true,\"filepath\":\"" + json_escape_str(filepath) + "\"}", "application/json");
        } else {
            res.status = 500;
            res.set_content("{\"error\":\"failed to write plugin\"}", "application/json");
        }
    });

    printf("=== AutoMatrix Server v0.3.0 ===\n");
    printf("Listening on http://localhost:%d\n", port);
    printf("Endpoints:\n");
    printf("  GET  /api/health\n");
    printf("  GET  /api/system\n");
    printf("  POST /api/model/load    {\"filepath\": \"...\"}\n");
    printf("  GET  /api/model/info\n");
    printf("  GET  /api/model/graph\n");
    printf("  GET  /api/devices\n");
    printf("  POST /api/models/scan   {\"directory\": \"...\"}\n");
    printf("  POST /api/deploy/push   {\"serial\": \"..\", \"binaryDir\": \"..\"}\n");
    printf("  POST /api/inference/run  {\"serial\": \"..\", \"modelPath\": \"..\"}\n");
    printf("================================\n");

    svr.listen("0.0.0.0", port);
    return 0;
}

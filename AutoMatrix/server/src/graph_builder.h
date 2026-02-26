#pragma once
#include <string>

// Build a visual graph representation from GGUF tensor names
// Returns JSON: { nodes: [...], edges: [...] }
std::string build_model_graph(const std::string & filepath);

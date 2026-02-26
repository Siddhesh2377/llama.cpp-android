#pragma once
#include <string>

// Parse a GGUF file and return JSON metadata
// Returns empty string on error
std::string parse_gguf(const std::string & filepath);

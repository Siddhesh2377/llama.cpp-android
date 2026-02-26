#pragma once
#include <string>

// Get JSON array of connected ADB devices
// Returns: {"devices": [...]}
std::string get_adb_devices();

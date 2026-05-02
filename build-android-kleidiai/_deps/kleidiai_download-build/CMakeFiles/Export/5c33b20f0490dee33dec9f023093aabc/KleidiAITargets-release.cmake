#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "KleidiAI::kleidiai" for configuration "Release"
set_property(TARGET KleidiAI::kleidiai APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(KleidiAI::kleidiai PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libkleidiai.so"
  IMPORTED_SONAME_RELEASE "libkleidiai.so"
  )

list(APPEND _cmake_import_check_targets KleidiAI::kleidiai )
list(APPEND _cmake_import_check_files_for_KleidiAI::kleidiai "${_IMPORT_PREFIX}/lib/libkleidiai.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)

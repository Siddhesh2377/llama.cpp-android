# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/home/dev/include/llama.cpp/build-android-kleidiai/_deps/kleidiai_download-src")
  file(MAKE_DIRECTORY "/home/home/dev/include/llama.cpp/build-android-kleidiai/_deps/kleidiai_download-src")
endif()
file(MAKE_DIRECTORY
  "/home/home/dev/include/llama.cpp/build-android-kleidiai/_deps/kleidiai_download-build"
  "/home/home/dev/include/llama.cpp/build-android-kleidiai/_deps/kleidiai_download-subbuild/kleidiai_download-populate-prefix"
  "/home/home/dev/include/llama.cpp/build-android-kleidiai/_deps/kleidiai_download-subbuild/kleidiai_download-populate-prefix/tmp"
  "/home/home/dev/include/llama.cpp/build-android-kleidiai/_deps/kleidiai_download-subbuild/kleidiai_download-populate-prefix/src/kleidiai_download-populate-stamp"
  "/home/home/dev/include/llama.cpp/build-android-kleidiai/_deps/kleidiai_download-subbuild/kleidiai_download-populate-prefix/src"
  "/home/home/dev/include/llama.cpp/build-android-kleidiai/_deps/kleidiai_download-subbuild/kleidiai_download-populate-prefix/src/kleidiai_download-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/home/dev/include/llama.cpp/build-android-kleidiai/_deps/kleidiai_download-subbuild/kleidiai_download-populate-prefix/src/kleidiai_download-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/home/dev/include/llama.cpp/build-android-kleidiai/_deps/kleidiai_download-subbuild/kleidiai_download-populate-prefix/src/kleidiai_download-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()

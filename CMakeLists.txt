cmake_minimum_required(VERSION 3.20)
project(recduplicate_detect VERSION 0.2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(RECDUP_ENABLE_FFMPEG "Build media decoding support with FFmpeg" ON)
option(RECDUP_BUILD_TESTS "Build unit tests" ON)

if(NOT DEFINED FFMPEG_ROOT AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ffmpegSHARED/include/libavformat/avformat.h")
  set(FFMPEG_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/ffmpegSHARED" CACHE PATH
      "FFmpeg SDK root" FORCE)
endif()

set(RECDUP_HAS_FFMPEG OFF)
if(RECDUP_ENABLE_FFMPEG)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(FFMPEG QUIET IMPORTED_TARGET
      libavformat libavcodec libavutil libswscale libswresample)
    if(FFMPEG_FOUND)
      set(RECDUP_HAS_FFMPEG ON)
      set(RECDUP_FFMPEG_TARGET PkgConfig::FFMPEG)
    endif()
  endif()

  if(NOT RECDUP_HAS_FFMPEG AND DEFINED FFMPEG_ROOT)
    find_path(FFMPEG_INCLUDE_DIR libavformat/avformat.h
      PATHS "${FFMPEG_ROOT}/include" NO_DEFAULT_PATH)
    foreach(lib avformat avcodec avutil swscale swresample)
      find_library(FFMPEG_${lib}_LIBRARY NAMES ${lib}
        PATHS "${FFMPEG_ROOT}/lib" "${FFMPEG_ROOT}/bin" NO_DEFAULT_PATH)
    endforeach()
    if(FFMPEG_INCLUDE_DIR AND FFMPEG_avformat_LIBRARY AND
       FFMPEG_avcodec_LIBRARY AND FFMPEG_avutil_LIBRARY AND
       FFMPEG_swscale_LIBRARY AND FFMPEG_swresample_LIBRARY)
      add_library(recdup_ffmpeg INTERFACE)
      target_include_directories(recdup_ffmpeg INTERFACE "${FFMPEG_INCLUDE_DIR}")
      target_link_libraries(recdup_ffmpeg INTERFACE
        "${FFMPEG_avformat_LIBRARY}" "${FFMPEG_avcodec_LIBRARY}"
        "${FFMPEG_avutil_LIBRARY}" "${FFMPEG_swscale_LIBRARY}"
        "${FFMPEG_swresample_LIBRARY}")
      set(RECDUP_FFMPEG_TARGET recdup_ffmpeg)
      set(RECDUP_HAS_FFMPEG ON)
    endif()
  endif()
endif()

add_library(recdup_core
  src/detector.cpp
  src/json_writer.cpp
  src/media_analyzer.cpp
  src/programme_inference.cpp
  src/timestamp_normalizer.cpp
  src/vector_database.cpp)
target_include_directories(recdup_core PUBLIC include)
target_compile_definitions(recdup_core PUBLIC
  RECDUP_HAS_FFMPEG=$<BOOL:${RECDUP_HAS_FFMPEG}>)
if(RECDUP_HAS_FFMPEG)
  target_link_libraries(recdup_core PRIVATE ${RECDUP_FFMPEG_TARGET})
else()
  message(WARNING "FFmpeg development libraries were not found. The CLI will build, but media analysis will be unavailable.")
endif()
if(MSVC)
  target_compile_options(recdup_core PRIVATE /W4 /permissive-)
else()
  target_compile_options(recdup_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(recdup src/main.cpp)
target_link_libraries(recdup PRIVATE recdup_core)
function(recdup_copy_ffmpeg_runtime target)
  if(RECDUP_HAS_FFMPEG AND WIN32 AND EXISTS "${FFMPEG_ROOT}/bin")
    add_custom_command(TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${FFMPEG_ROOT}/bin/avformat-62.dll"
      "${FFMPEG_ROOT}/bin/avcodec-62.dll"
      "${FFMPEG_ROOT}/bin/avutil-60.dll"
      "${FFMPEG_ROOT}/bin/swscale-9.dll"
      "${FFMPEG_ROOT}/bin/swresample-6.dll"
      "$<TARGET_FILE_DIR:${target}>")
  endif()
endfunction()
recdup_copy_ffmpeg_runtime(recdup)

function(recdup_copy_mingw_runtime target)
  if(MINGW)
    foreach(runtime libgcc_s_seh-1.dll libstdc++-6.dll libwinpthread-1.dll)
      execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=${runtime}
        OUTPUT_VARIABLE runtime_path OUTPUT_STRIP_TRAILING_WHITESPACE)
      if(NOT EXISTS "${runtime_path}")
        get_filename_component(compiler_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
        set(runtime_path "${compiler_dir}/${runtime}")
      endif()
      if(EXISTS "${runtime_path}")
        add_custom_command(TARGET ${target} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${runtime_path}" "$<TARGET_FILE_DIR:${target}>")
      endif()
    endforeach()
  endif()
endfunction()
recdup_copy_mingw_runtime(recdup)

include(GNUInstallDirs)
install(TARGETS recdup RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
if(RECDUP_HAS_FFMPEG AND WIN32 AND EXISTS "${FFMPEG_ROOT}/bin")
  install(FILES
    "${FFMPEG_ROOT}/bin/avformat-62.dll"
    "${FFMPEG_ROOT}/bin/avcodec-62.dll"
    "${FFMPEG_ROOT}/bin/avutil-60.dll"
    "${FFMPEG_ROOT}/bin/swscale-9.dll"
    "${FFMPEG_ROOT}/bin/swresample-6.dll"
    DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(RECDUP_BUILD_TESTS)
  enable_testing()
  add_executable(recdup_tests tests/test_core.cpp)
  target_link_libraries(recdup_tests PRIVATE recdup_core)
  recdup_copy_ffmpeg_runtime(recdup_tests)
  recdup_copy_mingw_runtime(recdup_tests)
  add_test(NAME recdup_core_tests COMMAND recdup_tests)
endif()

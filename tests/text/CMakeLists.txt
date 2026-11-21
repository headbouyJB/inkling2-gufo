# Tests for inkling2_text. Works inside the extension build or standalone:
#   cmake -S ext/tests/text -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build && ctest --test-dir build
cmake_minimum_required(VERSION 3.20)
if(NOT TARGET inkling2_text)
  project(inkling2_text_tests CXX)
  enable_testing()
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../src/text ${CMAKE_CURRENT_BINARY_DIR}/inkling2_text)
endif()

add_executable(inkling2_text_tests text_tests.cpp)
target_link_libraries(inkling2_text_tests PRIVATE inkling2_text)
target_compile_definitions(inkling2_text_tests PRIVATE
  INKLING2_TEXT_GOLDENS_DIR="${CMAKE_CURRENT_LIST_DIR}/goldens")
# The tokenizer is read from a GGUF: set INKLING2_TEST_GGUF (an Inkling 2 GGUF).
add_test(NAME inkling2_text COMMAND inkling2_text_tests)
set_tests_properties(inkling2_text PROPERTIES SKIP_RETURN_CODE 77)

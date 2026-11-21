# Inkling 2 Mini text front-end: tokenizer, chat template, tool dialect.
# HIP-free, no Gufo dependency (gufo_adapter.hpp is header-only glue that the
# runner includes). Add with add_subdirectory(src/text); include dir is src/.
add_library(inkling2_text STATIC
  chat_template.cpp
  gguf_meta.cpp
  json.cpp
  pretokenizer.cpp
  tokenizer.cpp
  tool_dialect.cpp
  unicode.cpp
  unicode_tables.cpp)
target_include_directories(inkling2_text PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_features(inkling2_text PUBLIC cxx_std_20)
set_target_properties(inkling2_text PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Inkling 2 Mini model extension.

# Host-only: configuration, probe and weight table. No HIP, so diagnostics and
# probes work on machines without a GPU.
gufo_extension_target(
  NAME gufo_inkling2
  SOURCES
    src/config.cpp
    src/weights.cpp
    src/probe.cpp
    src/extension.cpp
    src/text/chat_template.cpp
    src/text/gguf_meta.cpp
    src/text/json.cpp
    src/text/pretokenizer.cpp
    src/text/tokenizer.cpp
    src/text/tool_dialect.cpp
    src/text/unicode.cpp
    src/text/unicode_tables.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_ext_core
)
if(TARGET gufo_inkling2)
  # The text front-end includes its own headers as "text/...".
  target_include_directories(gufo_inkling2 PUBLIC ${GUFO_EXTENSION_BUILD_ROOT}/src)
endif()

# The runner and provider registration.
gufo_extension_target(
  NAME gufo_inkling2_hip
  SOURCES
    src/engine.cpp
    src/dspark.cpp
    src/state_layout.cpp
    src/cache_adapter.cpp
    src/runner.cpp
    src/provider.cpp
    src/vision/vision_image.cpp
    src/vision/mmproj_gguf.cpp
  LINK_LIBRARIES
    gufo_inkling2
    gufo_ext_runtime
    gufo_ext_core
    gufo_ext_hip
)
if(TARGET gufo_inkling2_hip)
  gufo_extension_hip_sources(gufo_inkling2_hip SOURCES src/hip/kernels.hip src/hip/fast.hip src/hip/dspark.hip src/vision/vision_encoder.hip)
endif()

# GPU tools (test role: built only with BUILD_TESTING=ON and HIP).
gufo_extension_target(
  NAME inkling2_oracle
  SOURCES
    tests/oracle/oracle.cpp
  LINK_LIBRARIES
    gufo_inkling2_hip
    gufo_inkling2
    gufo_ext_runtime
    gufo_ext_core
    gufo_ext_hip
)

# Capture/restore and layered-prefill bit-identity on real weights (GPU; takes GGUF + ids paths).
gufo_extension_target(
  NAME inkling2_cache_test
  SOURCES
    tests/cache/cache_test.cpp
  LINK_LIBRARIES
    gufo_inkling2_hip
    gufo_inkling2
    gufo_ext_runtime
    gufo_ext_core
    gufo_ext_hip
)

# Image prompt oracle (M5): renders an image prompt, encodes with the mmproj, injects the rows (GPU).
gufo_extension_target(
  NAME inkling2_image_oracle
  SOURCES
    tests/vision/image_oracle.cpp
  LINK_LIBRARIES
    gufo_inkling2_hip
    gufo_inkling2
    gufo_ext_runtime
    gufo_ext_core
    gufo_ext_hip
)

# Prompt-lookup speculation (0.1.3): runner-level agreement with plain decoding and the verify/rollback stress (GPU).
gufo_extension_target(
  NAME inkling2_lookup_test
  SOURCES
    tests/lookup/lookup_test.cpp
  LINK_LIBRARIES
    gufo_inkling2_hip
    gufo_inkling2
    gufo_ext_runtime
    gufo_ext_core
    gufo_ext_hip
)

# DSpark drafter (0.1.3 step 2): oracle comparison against llama.cpp dumps, acceptance probe, negative control, timing (GPU).
gufo_extension_target(
  NAME inkling2_dspark_test
  SOURCES
    tests/dspark/dspark_test.cpp
  LINK_LIBRARIES
    gufo_inkling2_hip
    gufo_inkling2
    gufo_ext_runtime
    gufo_ext_core
    gufo_ext_hip
)

# The test-only runner factory and chain hook (src/runner_test.hpp) are compiled into the runner library only when
# the lookup or the DSpark test is built (BUILD_TESTING); a module built without the tests has neither. The hooks are an
# unset std::function in normal use.
if((TARGET inkling2_lookup_test OR TARGET inkling2_dspark_test) AND TARGET gufo_inkling2_hip)
  target_compile_definitions(gufo_inkling2_hip PRIVATE INKLING2_TEST_HOOKS=1)
endif()

gufo_extension_target(
  NAME inkling2_bench
  SOURCES
    tests/bench/bench.cpp
  LINK_LIBRARIES
    gufo_inkling2_hip
    gufo_inkling2
    gufo_ext_runtime
    gufo_ext_core
    gufo_ext_hip
)

gufo_extension_target(
  NAME inkling2_probe_test
  SOURCES
    tests/probe_test.cpp
  LINK_LIBRARIES
    gufo_inkling2
    gufo_ext_runtime
    gufo_ext_core
)

# Adaptive lookup draft length (src/lookup_policy.hpp): pure CPU, no model.
gufo_extension_target(
  NAME inkling2_policy_test
  SOURCES
    tests/lookup/policy_test.cpp
  LINK_LIBRARIES
    gufo_inkling2
    gufo_ext_runtime
    gufo_ext_core
)

# Takes GGUF paths on the command line; without any it skips.
gufo_extension_target(
  NAME inkling2_weights_test
  SOURCES
    tests/weights_test.cpp
  LINK_LIBRARIES
    gufo_inkling2
    gufo_ext_runtime
    gufo_ext_core
)

if(TARGET inkling2_probe_test)
  add_test(NAME inkling2.probe COMMAND inkling2_probe_test)
  add_test(NAME inkling2.weights COMMAND inkling2_weights_test)
endif()
if(TARGET inkling2_policy_test)
  add_test(NAME inkling2.lookup_policy COMMAND inkling2_policy_test)
endif()

install(FILES "${GUFO_EXTENSION_BUILD_ROOT}/LICENSE"
  "${GUFO_EXTENSION_BUILD_ROOT}/THIRD_PARTY.md"
  DESTINATION share/licenses/gufo/extensions/inkling2)

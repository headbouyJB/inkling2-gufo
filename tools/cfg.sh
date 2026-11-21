#!/bin/bash
# configure the HIP dev build (one-off). Build dir: $WORK_ROOT/ext-hip
./dev-env.sh "cmake --preset release -B /work/ext-hip -GNinja -DBUILD_TESTING=ON \
 -DGUFO_EXTENSION_DIRS=/work/ext-build \
 -DCMAKE_HIP_COMPILER=/nix/store/92bpbb2rmrkszhw83x63x7qzj9jsz324-rocm-toolchain/bin/clang \
 -DGUFO_NPU_PREBUILT_DESIGNS=/nix/store/5h1nwjlxqpjgqnix7yijrrkvlbzqzb0n-gufo-npu-designs/share/gufo/npu \
 -DXRT_DIR=/nix/store/f1sqyi01ryvc9gvmdlmb5m2h01p1c6f6-xrt-amdxdna-2.26-06c43a7/opt/xilinx/xrt/share/cmake/XRT \
 -DGUFO_FFMPEG_EXECUTABLE=/nix/store/fqpym60z9jyv2a7cal8411axw7jbknxh-ffmpeg-headless-9.0.2-bin/bin/ffmpeg \
 -DGUFO_FFPROBE_EXECUTABLE=/nix/store/fqpym60z9jyv2a7cal8411axw7jbknxh-ffmpeg-headless-9.0.2-bin/bin/ffprobe \
 -DGUFO_RELEASE_VERSION=development -DGUFO_REVISION=0260e11"

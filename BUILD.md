# Building and testing

Target: Gufo v0.35.0 (`0260e11`), extension API 0.3.x (the v0.35.0 build
reports 0.3.20). Verified 2026-11-21.

The commands below use these environment variables. Set them for your machine
(the scripts in `tools/` and `tests/vision/` read the same names and have
defaults under `$HOME`):

| Variable | Meaning |
| --- | --- |
| `GUFO_SRC` | checkout of Gufo v0.35.0 (`0260e11`); the container commands expect it at `$WORK_ROOT/gufo-src` (default) |
| `WORK_ROOT` | scratch directory, mounted at `/work` in the container (`ext-build/`, `ext-cpu/`, `ext-hip/`, `m1/`, ... live here) |
| `MODEL_DIR` | directory holding `inkling2-mini/` (GGUF files), mounted read-only at `/models` |
| `CORPUS_DIR` | directory with `wiki.test.raw` (WikiText-2 test split) for the oracle and smoke scripts |
| `LLAMA_CPP_SRC` | llama.cpp checkout at e412c6f plus PR #32704 (only for the oracle tools; `tests/vision/build_oracle.sh` defaults to `$HOME/src/llama.cpp`) |
| `ROCM_PATH` | TheRock 7.14 layout only (`$ROCM_PATH/opt/rocm/core-7.14/bin` is prepended to `PATH`); otherwise `hipcc` is taken from `PATH`. Used by `tests/vision/build.sh`, `lab/build.sh`, `lab/attn/build.sh`; default `/opt/rocm-therock-7.14` |
| `CXX` | host C++ compiler for `tests/vision/build_oracle.sh`; default `/usr/lib/llvm-21/bin/clang++` |
| `EXT_SRC` | optional, only for `tools/build.sh`: if set, the extension tree (this repository's root) it re-syncs to `$WORK_ROOT/ext-build/` before building; if unset, it builds `ext-build/` as it is |
| `GPU_RUN` | optional wrapper `GPU_RUN <label> <cmd...>` to serialise GPU jobs; defaults to `tools/nowrap.sh`, which just runs the command |
| `FCLIENT_VENV` | Python venv used for Gufo's functional-test client (only for `functional*.sh`) |

This repository is the extension (its root holds `extension.json`, `extension.cmake` and `src/`). Copy it to
the build host (`lab/` is not part of the build; `flake.lock` is kept on the host). Run this from the repository root:

```sh
rsync -a --delete --exclude .git --exclude lab --exclude flake.lock \
  ./ build-host:$WORK_ROOT/ext-build/
```

Drop `build-host:` when building on the same machine; `$WORK_ROOT` here is expanded on the machine you run
`rsync` on, so write out the build machine's path if it differs.

### Toolchains

- The extension itself (sections 1 to 3 below) builds through Gufo's Nix flake: `flake.lock` pins `nixpkgs-rocm10` (gufo-org/nixpkgs, rev `1a6ba8b`), and `docs/dev/M3-decode.md` records a ROCm 10.1 toolchain for the measurements. The exact ROCm version inside that pin was not re-checked from `flake.lock` alone.
- The kernel lab (`lab/`), the vision oracle tools (`tests/vision/build.sh`, `build_oracle.sh`) and the llama.cpp baseline builds used a host ROCm 7.14 (TheRock) install, with llvm-21 clang for the C++ oracle tools (`docs/dev/llama-cpp-baseline.md`).

### Helper scripts

The scripts in `tools/` are meant to be copied to `$WORK_ROOT/m1/` (`cp tools/* $WORK_ROOT/m1/`). They `cd` there and call their siblings
through `$W` (default `$WORK_ROOT/m1`): `gufo-wrap.sh`, `run-gpu.sh`, `dev-env.sh`, `disk_client.py` and `sessions_client.py`. `tests/vision/run_compare.sh` also runs from `$W` and expects `$W/run-gpu.sh`, `$W/tools/mtmd_logits` (built by `tests/vision/build_oracle.sh $W/tools`), `$W/img/` (from `tests/vision/gen_images.py`) and `$WORK_ROOT/ext-build/tools/compare_dumps.py`.

All commands below run on the build machine, CPU only, in the usual
Nix-in-podman container (`c` abbreviates the podman prefix; a shell function, because
a variable would not keep the quoting of `NIX_CONFIG`):

```sh
c() {
  nice -n 10 podman run --rm --security-opt label=disable \
    -e NIX_CONFIG=$'experimental-features = nix-command flakes\nsandbox = false' \
    -v gufo-nix:/nix -v $WORK_ROOT:/work "$@"
}
```

## 0. Install as a module (primary route for users)

```sh
gufo extension install https://github.com/headbouyJB/inkling2-gufo@<commit> --yes
gufo extension list      # expect a line like: inkling2 0.1.1 [installed, ready]
```

`<commit>` is the release commit (the full hash is safest; Gufo accepts any commit or tag). Gufo builds the module on your machine from that pinned source
and keeps it under `$GUFO_EXTENSIONS_DIR` (default `~/.local/share/gufo/extensions`).

`--yes` skips Gufo's confirmation prompt for unreviewed sources; read the source first (it is native code that runs with your permissions).

Requirement: a Gufo build that ships the extension SDK. That is the `sdk/` directory of a source build tree
(`<build>/sdk`, beside the `gufo` binary), `<prefix>/lib/gufo/sdk` after installing such a build, or any SDK you
point Gufo at with `GUFO_SDK_DIR`. The SDK has to come from the same Gufo build as the `gufo` binary you run. You also need the
toolchain that built that Gufo: CMake 3.27 or newer, Ninja, the same C, C++ and HIP compilers, the ROCm libraries, and
Git. A Gufo without the SDK (for example the default Nix package) cannot build modules. Gufo's docs name the Nix `#sdk` / `#release-sdk` outputs and the `gufo-runtime` container as SDK-carrying alternatives; neither was tried with this extension. Otherwise use the developer routes below (sections 1 to 3).

Verified: builds and installs as a module on a plain Gufo v0.35.0 source build with `GUFO_SDK_DIR=<build>/sdk`. The measurements in the README come from the in-tree build. Gufo marks modules as experimental (as of November 2026), so the commands may change.
Served on the GPU (gfx1151) from the installed 0.1.1 module, with no in-tree copy built in: chat, concurrent requests, tool calls, image input, `UD-Q8_K_XL`, and Gufo's discovery, responses, stops, conversation, state-edges, prefill-scheduling and cache functional suites pass; four fixed prompts at temperature 0 give the same text as the in-tree build; and in two Gufo `compare.py` timing pairs no measurement was marked slower than the in-tree build (0 failed; 16 and 5 inconclusive), though neither pair qualified, because one disk-restore cache case varies from run to run on both builds. Details in [`docs/dev/module-test.md`](docs/dev/module-test.md).

After a Gufo upgrade `gufo extension list` shows
`needs update` until you run `gufo extension update`.

Sections 1 to 3 are the developer route: the extension is compiled into a Gufo build tree that you control.

## 1. Unit tests (CPU preset, `GUFO_EXTENSION_DIRS`)

```sh
c -w /work/gufo-src docker.io/nixos/nix:latest sh -c '
  mkdir -p /build && chmod 1777 /build
  git config --global --add safe.directory "*"
  nix develop --accept-flake-config -c sh -c "
    cmake --preset cpu-test -B /work/ext-cpu \
      -DGUFO_EXTENSION_DIRS=/work/ext-build &&
    cmake --build /work/ext-cpu \
      --target inkling2_probe_test inkling2_weights_test -j 16 &&
    ctest --test-dir /work/ext-cpu -R inkling2 --output-on-failure"'
```

The text front-end (tokenizer, chat template, tool dialect, stream decoder) has its own test
program, which builds standalone with any C++20 compiler and needs one Inkling 2 GGUF for the tokenizer:

```sh
cmake -S tests/text -B text-build -GNinja -DCMAKE_BUILD_TYPE=Release && cmake --build text-build
INKLING2_TEST_GGUF=$MODEL_DIR/inkling2-mini/unsloth/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file \
  ctest --test-dir text-build --output-on-failure      # "PASS: 8354 checks, 0 failed" with -V
```

`inkling2.probe` runs on synthetic GGUF headers. `inkling2.weights` takes
files on its command line and skips without any. Run it on real files with the
binary the build leaves under `/work/ext-cpu/extensions/build/inkling2/build/`:

```sh
T=$WORK_ROOT/ext-cpu/extensions/build/inkling2/build/inkling2_weights_test
M=$MODEL_DIR/inkling2-mini/unsloth
$T --equal $M/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file \
           $M/Inkling-2-Mini-UD-Q4_K_XL.gguf
```

## 2. Full `gufo` with the extension (documented Nix route)

`flake.nix` pins Gufo `0260e11` and calls `override { extensions = [ self ]; }`.

```sh
cd $WORK_ROOT && c -w /work/ext-build docker.io/nixos/nix:latest sh -c '
  git config --global --add safe.directory "*"
  nix build -L --out-link /work/ext-nix-result'
```

This compiles all of Gufo with HIP (about 20 minutes on the build machine). Result:
`/nix/store/llpxjdr67lwg8l80rdg97yndxyd8lv2w-gufo-0-unstable-0260e11`, inside
the `gufo-nix` volume, so run it in the same container. No GPU device is given
to the container, so nothing touches the GPU:

```sh
G=/nix/store/llpxjdr67lwg8l80rdg97yndxyd8lv2w-gufo-0-unstable-0260e11/bin/gufo
podman run --rm --security-opt label=disable -v gufo-nix:/nix:ro \
  -v $MODEL_DIR:/models:ro docker.io/nixos/nix:latest $G diagnose --section extensions
podman run ... $G serve --model /models/inkling2-mini/unsloth/Inkling-2-Mini-UD-Q4_K_XL.gguf --port 18999
```

With no GPU device in the container, `gufo diagnose` shows that the extension is registered and `gufo serve`
only checks that a file is accepted (a refusal reads "No text provider found for model architecture"). The reason
for a refusal is printed by `inkling2_weights_test FILE`. To run the model, give the container the GPU
(see section 3 and README.md).

## Differences between v0.35.0 and main `029a156`

The extension API surface used here is the same. Main adds
`ProviderInfo::hardware_queues` (speech providers only, defaults to 0) and
`serving.hardware_queues` in the schema, more `changes/0.3/*` records (API
0.3.35 against 0.3.20), `experimental/state_memory.hpp`, and `gufo_json` in the
extension host libraries. None of it is needed or conflicts. Main has not been
built with this extension.

One v0.35.0 rule worth knowing: an NVFP4 tensor reaches the provider unsized
(`accepts_unsized_tensors = true`; the extension sizes it: 36 bytes per 64
values). The manifest declares no speculative-decoding mode and no extra
options, because no drafter exists; if one is added later, `serving.draft_options`
must name an option the extension declares in `options`.

## 3. Fast dev loop (HIP, incremental): `nix develop` + a persistent CMake tree

The full `nix build` above recompiles all of Gufo (about 20 minutes). For development configure
the Gufo source tree once inside the same Nix devshell, with the extension attached through
`GUFO_EXTENSION_DIRS`, and let Ninja rebuild only what changed. One configure plus the first full
build took about 10 minutes on the build machine; after that an edit to the extension costs 20 to 60 seconds
(the HIP object, the extension archives and the final `gufo` link), and a rebuild of just the
GPU tool `inkling2_oracle` about 20 seconds. The scripts are in `tools/`; paths assume
`$WORK_ROOT/m1` on the build machine (the `tools/` copy above); the extension is synced to `$WORK_ROOT/ext-build/` (`tools/build.sh` re-syncs it from `$EXT_SRC` when that is set).

Sync the extension to `$WORK_ROOT/ext-build/` (the `rsync` below, or the one of the introduction) **before** the configure: with an empty
`ext-build/` the configure succeeds but the extension's targets do not exist (`ninja: error: unknown target 'inkling2_oracle'`).

```sh
# Set once, in the shell you run the commands below from. REPO must be the absolute path of this
# repository's root (the directory holding extension.json). Never rsync from any other directory.
REPO=/absolute/path/to/this/repository

# 1. On your machine, from any directory: sync the extension first. Drop `build-host:` if you build on
#    this machine. $WORK_ROOT is expanded on your machine, so write out the build machine's path if it
#    differs. The :? guard stops the command if REPO is unset (an empty REPO would make the source "/").
rsync -a --delete --exclude .git --exclude lab --exclude flake.lock "${REPO:?set REPO to this repository root}"/ \
  build-host:$WORK_ROOT/ext-build/

# 2. On the build machine, in $WORK_ROOT/m1 (where dev-env.sh and cfg.sh live): configure once.
#    dev-env.sh "<cmd>": `nix develop` of $GUFO_SRC (v0.35.0) inside the usual podman container
#    (no GPU device: building never touches the GPU).
cd $WORK_ROOT/m1
./dev-env.sh "cmake --preset release -B /work/ext-hip -GNinja -DBUILD_TESTING=ON \
  -DGUFO_EXTENSION_DIRS=/work/ext-build \
  -DCMAKE_HIP_COMPILER=/nix/store/92bpbb2rmrkszhw83x63x7qzj9jsz324-rocm-toolchain/bin/clang \
  -DGUFO_NPU_PREBUILT_DESIGNS=/nix/store/5h1nwjlxqpjgqnix7yijrrkvlbzqzb0n-gufo-npu-designs/share/gufo/npu \
  -DXRT_DIR=<xrt-amdxdna>/opt/xilinx/xrt/share/cmake/XRT \
  -DGUFO_FFMPEG_EXECUTABLE=<ffmpeg>/bin/ffmpeg -DGUFO_FFPROBE_EXECUTABLE=<ffprobe> \
  -DGUFO_RELEASE_VERSION=development -DGUFO_REVISION=0260e11"      # (cfg.sh: the exact store paths)

# 3. On the build machine, in $WORK_ROOT/m1: build (repeat steps 1 and 3 after each edit).
./dev-env.sh "ninja -C /work/ext-hip -j 14 inkling2_oracle gufo"
```

To let `tools/build.sh` do step 1 for you, set `EXT_SRC` to the **absolute** path of this repository's root on the build machine; the script changes into `$WORK_ROOT/m1` before it syncs, so a relative path would not resolve.

The store paths come from the `cmake flags:` line of the `nix build -L` output (the command above does not write a log file; add `2>&1 | tee ext-nix-build.log` to keep one);
`BUILD_TESTING=ON` is only needed for the test-role GPU tool. Binaries land in
`/work/ext-hip/gufo` and `.../extensions/build/inkling2/build/inkling2_oracle`. Both live
in the `gufo-nix` volume's closure, so run them in the same container, now with the GPU:

```sh
podman run --rm --security-opt label=disable --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --ulimit memlock=-1:-1 --network host -v gufo-nix:/nix -v $WORK_ROOT:/work -v $MODEL_DIR:/models:ro \
  docker.io/nixos/nix:latest /work/ext-hip/gufo serve llm --model ... -p 5830
```

Inkling work shares the GPU: wrap such a command in `$GPU_RUN <label> ...` (optional, default runs the command directly) (see `tools/run-gpu.sh`).
Alternative that avoids a Gufo build tree altogether: Gufo 0.35 ships an extension SDK
(`gufo extension install .`, "Modules" in docs/extensions.md), but the SDK output of the
stock `0260e11` package was not realised on the build machine and the SDK of the extension-enabled build
already contains this extension, so we did not use it.

### Oracle and serve smoke
- `tests/oracle/oracle.cpp` (target `inkling2_oracle`): teacher-forces token ids through the
  engine and dumps top-64 + logsumexp per position; `tests/oracle/llama_dump.cpp` does the same
  through libllama (build it with `g++ ... -lllama`, CPU build of `$LLAMA_CPP_SRC`);
  `tools/compare_dumps.py` reports KL, top-1 agreement and per-range numbers.
- `tools/serve-inner.sh` + `tools/smoke.py`: start `gufo serve llm` on 127.0.0.1:5830 in a GPU
  container, run a script against it, stop it. Run under `$GPU_RUN`.

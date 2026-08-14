# Psyduck

making GPUs go fast from scratch

## Run the matrix-add GPU validation

The validation builds the Rust binary in a CUDA container, runs it on an NVIDIA T4,
and compares the GPU result with the same addition performed by the naive CPU
backend. A successful run validates the complete Rust-to-CUDA path: device
allocation, host-to-device upload, kernel launch, CUDA event capture,
device-to-host download, and result parity.

### One-time setup

Create a [Modal](https://modal.com/) account and install
[just](https://just.systems/). From the repository root, install the pinned
Python and Modal tooling:

```shell
just configure-modal
```

Then authenticate the Modal CLI:

```shell
deployment/modal/.tools/bin/uv run \
  --project deployment/modal \
  --locked \
  modal token new
```

Alternatively, copy `.env.example` to `.env` and provide
`PSYDUCK_MODAL_TOKEN_ID` and `PSYDUCK_MODAL_TOKEN_SECRET`. The `.env` file is
gitignored and is not copied into the Modal image. The `justfile` maps these
credentials to Modal's standard environment only for the local invocation.

### Run the validation

```shell
just modal-matrix-add-validate
```

The first run may take longer while Modal builds the image. The command exits
successfully only when the GPU and CPU results match. Look for output resembling:

```json
{
  "validation": {
    "actual": [6.0, 8.0, 10.0, 12.0],
    "expected": [6.0, 8.0, 10.0, 12.0],
    "gpu_elapsed_us": 19.136,
    "status": "passed"
  },
  "exit_code": 0,
  "gpu": "T4",
  "rust_binary": "gpu_add_validate"
}
```

The elapsed time is a CUDA-event measurement around one `2 x 2` addition. It
confirms that timing capture works, but it is dominated by launch overhead and
must not be treated as a kernel-throughput benchmark.

The recipe pins the validation to an NVIDIA T4 and Rust `1.96.0`. All Psyduck-specific
Modal configuration uses the `PSYDUCK_MODAL_` prefix.

### Other commands

Run another Rust binary with `just modal-run <binary> <gpu>`. Re-synchronize
or verify the Python tooling with:

```shell
just modal-sync
just modal-check
```

# Plan

1. Validate add / mul on Nvidia GPU via rust
2. Get running on CPU level
    a) naive
    b) with SIMD / striding (need to understand why striding doesn't solve the problem in full)
    c) gather charts to find saturation point
3. Move to GPU implementation of matmul
4. Once saturated, impl KvCache

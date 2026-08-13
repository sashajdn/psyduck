# Psyduck

making GPUs go fast from scratch

## Run matrix add on Modal

Install [just](https://just.systems/), configure the pinned local Python
tooling, then authenticate with Modal:

```shell
just configure-modal
deployment/modal/.tools/bin/uv run \
  --project deployment/modal \
  --locked \
  modal token new
just modal-matrix-add
```

Alternatively, copy `.env.example` to `.env` and provide
`PSYDUCK_MODAL_TOKEN_ID` and `PSYDUCK_MODAL_TOKEN_SECRET`. The `justfile` maps
those credentials to Modal's standard environment at invocation time. All
Psyduck-specific Modal configuration uses the `PSYDUCK_MODAL_` prefix.
The Modal recipes pin the GPU to an NVIDIA L4 and the Rust toolchain to
`1.96.0` in the `justfile`.

Run another Rust binary with `just modal-run <binary>`. Re-synchronize or
verify the Python tooling with:

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

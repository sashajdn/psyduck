# Psyduck

Making GPUs go fast from scratch


## Tracking

### Square Matrices

These tables track square `f32` matmul by optimization and commit. Throughput
uses the `2*N³` FLOP convention; kernel time is the total for 10 measured
operations and excludes warmups and harness work. For variants that prepare
data inside `try_matmul`, that preparation is included.

> The square-matrix benchmark performs three warmup operations before collecting 10 measured operations.

#### Aggregate throughput (GFLOP/s)

| Optimization | Target | Commit | N=4 | N=8 | N=16 | N=32 | N=64 | N=128 | N=256 | N=512 | N=1024 | N=2048 | N=4096 |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| naive | CPU | [`bdfa68b04359`](https://github.com/sashajdn/psyduck/commit/bdfa68b04359733820164f95f76c8069303da405)\* | 0.355 | 0.638 | 0.709 | 0.740 | 0.746 | 1.058 | 0.686 | 0.594 | 0.563 | 0.207 | ❌ |
| transposed_b | CPU | [`ebabe625918c`](https://github.com/sashajdn/psyduck/commit/ebabe625918c5451342b973d91861352030fbde9)\* | 0.346 | 0.806 | 1.156 | 1.404 | 1.607 | 1.575 | 1.659 | 1.820 | 1.803 | 1.756 | 1.748 |

#### Kernel time (seconds)

| Optimization | Target | Commit | N=4 | N=8 | N=16 | N=32 | N=64 | N=128 | N=256 | N=512 | N=1024 | N=2048 | N=4096 |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| naive | CPU | `bdfa68b04359`\* | 0.000004 | 0.000016 | 0.000115 | 0.000885 | 0.007032 | 0.039630 | 0.489186 | 4.522000 | 38.167563 | 831.913122 | ❌ |
| transposed_b | CPU | `ebabe625918c`\* | 0.000004 | 0.000013 | 0.000071 | 0.000467 | 0.003263 | 0.026632 | 0.202230 | 1.474606 | 11.912170 | 97.847109 | 786.297922 |


#### Relative to the naive baseline

Throughput is `optimization / naive`, so values above `1.00×` are faster.

| Optimization | Target | Commit | N=4 | N=8 | N=16 | N=32 | N=64 | N=128 | N=256 | N=512 | N=1024 | N=2048 | N=4096 |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| naive | CPU | `bdfa68b04359`\* | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | — |
| transposed_b | CPU | `ebabe625918c`\* | 0.98× | 1.26× | 1.63× | 1.90× | 2.15× | 1.49× | 2.42× | 3.06× | 3.20× | 8.48× | — |

| Optimization | Target | Commit | Throughput ↑ | Cycles/add ↓ | Instructions/add ↓ | L1 misses/add ↓ | L2 misses/add ↓ | L3 misses/add ↓ | Memory-stall cycles/add ↓ |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| naive | CPU | `bdfa68b04359`\* | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× |
| transposed_b | CPU | `ebabe625918c`\* | 3.64× | 3.67× | 5.47× | 534× | 2,105× | 857× | 2.78× |

#### Largest timing checkpoints

| Variant | N | FLOPs/operation | Seconds/operation | Throughput |
|:--|--:|--:|--:|--:|
| Naive | 2048 | 17.180 billion | 83.191 | 0.207 GFLOP/s |
| Transposed B | 4096 | 137.439 billion | 78.630 | 1.748 GFLOP/s |

## Run the matrix-add GPU validation

The validation builds the Rust binary in a CUDA container, runs it on an NVIDIA T4,
and compares the GPU result with the same addition performed by the naive CPU
backend. A successful run validates the complete Rust-to-CUDA path: device
allocation, host-to-device upload, kernel launch, CUDA event capture,
device-to-host download, and result parity.

## Optimizations

### CPU

#### Transposed

- Naive case is memory bound: cache misses per hierarchy increaes over some N boundary, proxied to given cache size & B-stride size.
- Transpose the `B` matrix for memory locality of the B-stride per `K`.
- Improved cache-line utilization, packs 16 f32 per 64B cacheline from the K length stride per iteration.
    - 1/16 in the naive case -> 16/16
- Reduces probability of memory reads, for given B elements, from fetching from higher & more costly caches.
- Increased computation cost initially, more memory locality improvements far exceeds this as N grows.
    - B^T costs ~`O(K * M)`, whereas naive reads are O(M * N * K).

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

## Run host matmul on a remote Linux machine

The remote workflow publishes the current tracked and non-ignored working tree
to immutable releases under the remote user's home directory. Local `.env`,
`.git`, and build artifacts are not transferred. A completed synchronization
atomically updates `~/psyduck/current`, while the Cargo target directory remains
shared across releases for incremental builds.

Configure a key-only SSH host in the local `.env` file:

```dotenv
PSYDUCK_REMOTE_HOST=psyduck-scaleway
PSYDUCK_REMOTE_DIR=psyduck
```

The remote host must provide Rust `1.96.0`, `just`, `rsync`, `perf`, and
`taskset`. Publish the current working tree without running it:

```shell
just remote-sync
```

Run the existing host matmul flow remotely. This synchronizes first and streams
the benchmark report back to the local terminal:

```shell
just matmul remote 1024 "" "" "" 5 3 1
```

For an interactive performance-counter run, synchronize locally and then enter
the machine:

```shell
just remote-sync
ssh psyduck-scaleway
cd ~/psyduck/current
just matmul-perf 1024 2 "" "" "" 5 3 1
```

The second argument is the `taskset` CPU list, so the example pins the process
to logical CPU 2. `matmul-perf` builds the release binary before starting
`perf stat`; compilation is therefore excluded from the counters. The counters
cover the entire binary process—including input generation, warmups, output
validation, and reporting—rather than only the internally sampled matmul
operations.

# Plan

1. Validate add / mul on Nvidia GPU via rust
2. Get running on CPU level
    a) naive
    b) with SIMD / striding (need to understand why striding doesn't solve the problem in full)
    c) gather charts to find saturation point
3. Move to GPU implementation of matmul
4. Once saturated, impl KvCache

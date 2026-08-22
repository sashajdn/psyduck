# Psyduck

Making GPUs go fast from scratch


## Tracking

### Square Matrices

This table tracks aggregate throughput and measured kernel time for square
`f32` matmul by commit. Throughput uses the `2*N³` FLOP convention; kernel time
is the total for 10 measured operations and excludes warmups and harness work.
For variants that prepare data inside `try_matmul`, that preparation is included.


| Note | Target | Commit | Measurement | N=4 | N=8 | N=16 | N=32 | N=64 | N=128 | N=256 | N=512 | N=1024 | N=2048 | N=4096 |
|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| naive_cpu | CPU | [`bdfa68b04359`](https://github.com/sashajdn/psyduck/commit/bdfa68b04359733820164f95f76c8069303da405)\* | Aggregate throughput (GFLOP/s) | 0.355 | 0.638 | 0.709 | 0.740 | 0.746 | 1.058 | 0.686 | 0.594 | 0.563 | 0.207 | ❌ |
| naive_cpu | CPU | `bdfa68b04359`\* | Kernel time (s) | 0.000004 | 0.000016 | 0.000115 | 0.000885 | 0.007032 | 0.039630 | 0.489186 | 4.522000 | 38.167563 | 831.913122 | ❌ |
| transpose_b_cpu | CPU | [`49f9e6e2b63c`](https://github.com/sashajdn/psyduck/commit/49f9e6e2b63c5c4c1973d43a307b75006b8cf468)\* | Aggregate throughput (GFLOP/s) | 0.346 | 0.806 | 1.156 | 1.404 | 1.607 | 1.575 | 1.659 | 1.820 | 1.803 | 1.756 | 1.748 |
| transpose_b_cpu | CPU | `49f9e6e2b63c`\* | Kernel time (s) | 0.000004 | 0.000013 | 0.000071 | 0.000467 | 0.003263 | 0.026632 | 0.202230 | 1.474606 | 11.912170 | 97.847109 | 786.297922 |
| transpose_b_cpu / naive_cpu | CPU | comparison | Throughput speedup | 0.98× | 1.26× | 1.63× | 1.90× | 2.15× | 1.49× | 2.42× | 3.06× | 3.20× | 8.48× | — |

`transpose_b_cpu` clones and transposes `B` inside every measured operation. Its
throughput remains approximately 1.75–1.82 GFLOP/s from `N=512` through
`N=4096`. Matched performance counters show that the plateau is backend-bound,
with TMA attributing a larger share to Memory bound than Core bound. This memory
category includes the on-core cache and load/store subsystem and does not imply
frequent DRAM access.

\* The benchmark report recorded a dirty working tree at the displayed base
commit.

> The square-matrix benchmark performs three warmup operations before collecting 10 measured operations.

### Matched CPU counter checkpoint

This comparison ran both implementations at `N=1024` with identical settings:
logical CPU 2, one warmup, two individually sampled operations, and three
process-level `perf` repetitions. Every raw counter had 100% coverage.

| Metric | Naive | Transposed B | Difference |
|:--|--:|--:|--:|
| Aggregate throughput | 0.494 GFLOP/s | 1.801 GFLOP/s | 3.64× higher |
| Cycles/add | 10.9871 | 2.9943 | 3.67× lower |
| Instructions/add | 21.0286 | 3.8479 | 5.47× lower |
| Cycles/instruction | 0.5225 | 0.7782 | 1.49× higher |
| L1 misses/add | 0.905116 | 0.001694 | 534× lower |
| L2 misses/add | 0.904271 | 0.000430 | 2,105× lower |
| L3 misses/add | 0.004931 | 0.00000575 | 857× lower |
| Any-memory stall cycles/add | 2.9711 | 1.0684 | 2.78× lower |

The matching `N=512` TMA runs used the same one-warmup/two-operation
configuration. These multiplexed estimates are directional:

| TMA metric | Naive | Transposed B |
|:--|--:|--:|
| Retiring | 44.1% | 31.2% |
| Frontend bound | 5.7% | 1.8% |
| Backend bound | 50.1% | 66.3% |
| Bad speculation | 0.2% | 0.2% |
| Core bound | 32.4% | 23.1% |
| Memory bound | 17.4% | 43.9% |

### Largest timing checkpoints

These are not shape-matched. They show the largest recorded timing point for
each implementation while keeping a single operation below five minutes:

| Variant | N | FLOPs/operation | Seconds/operation | Throughput |
|:--|--:|--:|--:|--:|
| Naive | 2048 | 17.180 billion | 83.191 | 0.207 GFLOP/s |
| Transposed B | 4096 | 137.439 billion | 78.630 | 1.748 GFLOP/s |

The transposed case performs 8× as much algorithmic work in slightly less wall
time. This is an equal-time frontier comparison, not an apples-to-apples shape
comparison.

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

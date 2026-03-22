# `#pragma unroll` Benchmark

## What this measures

A 10-instruction loop body with `rsqrtf` (SFU, ~16 cycle latency) at position 2,
followed by 8 dependent ALU ops. We sweep `#pragma unroll` from 1 (no unroll) to 16
and measure wall time.

## Environment

| | |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 (x8) |
| Compute capability | sm_90 |
| Driver | 535.230.02 |
| CUDA toolkit / nvcc | 12.2 (V12.2.140) |
| Host compiler | gcc 9.4.0 (used by nvcc for non-CUDA C++ only) |
| Kernel | Linux 5.15.0-1083-gcp |
| Compile flags | `-O3 -arch=sm_90 -std=c++17 --ptxas-options=-v` |

## Launch config

- 256 threads/block, 1024 blocks (262144 threads total)
- Each thread processes N floats from its own contiguous slice
- N=64 (64 MB, L1-resident) and N=512 (512 MB, L2-resident)

## Results

### N=64 (L1 resident — isolates compute)

| Kernel | Registers | Time (us) | Speedup |
|--------|-----------|-----------|---------|
| unroll_1 | 15 | 111.10 | 1.00x |
| unroll_2 | 20 | 78.88 | 1.41x |
| unroll_4 | 20 | 72.65 | 1.53x |
| unroll_8 | 30 | 71.13 | 1.56x |
| unroll_16 | 27 | 70.69 | 1.57x |

### N=512 (L2 resident — memory + compute)

| Kernel | Registers | Time (us) | Speedup |
|--------|-----------|-----------|---------|
| unroll_1 | 15 | 2111.57 | 1.00x |
| unroll_2 | 20 | 852.03 | 2.48x |
| unroll_4 | 20 | 541.73 | 3.90x |
| unroll_8 | 30 | 538.56 | 3.92x |
| unroll_16 | 27 | 536.66 | 3.93x |

## Key observations

1. **Biggest jump is 1→2.** Branch overhead eliminated + first opportunity for the
   scheduler to overlap SFU/load latencies across iterations.

2. **Diminishing returns after 4.** By unroll 4 the scheduler has enough independent
   instructions to keep the SFU and load units busy. Further unrolling adds register
   pressure (unroll_8 peaks at 30 regs) for negligible gain.

3. **N=512 amplifies the effect.** The 1→2 jump is 2.48x (vs 1.41x at N=64) because
   unrolling also overlaps memory latency from L2 misses — loads from iteration N+1
   are issued while iteration N's ALU chain runs.

4. **ptxas reorders across unrolled iterations.** SASS shows `MUFU.RSQ` calls from
   different iterations grouped together, not mechanically duplicated in sequence.
   The compiler is smarter than "copy-paste the loop body."

5. **Register pressure is non-monotonic.** unroll_8 uses 30 registers while unroll_16
   uses only 27 — the compiler's register allocator makes different tradeoffs at
   different unroll factors.

## Files

| File | Purpose |
|------|---------|
| `benchmark.cu` | All kernels + timing harness |
| `Makefile` | Build, PTX, SASS targets |
| `run.sh` | ncu profiling command |

## Usage

```bash
make            # build (register counts printed)
./benchmark     # run timing + correctness
make sass       # dump SASS for inspection
make ptx        # dump PTX
./run.sh        # ncu profiling (may need sudo)
```

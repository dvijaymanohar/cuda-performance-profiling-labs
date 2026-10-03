# Profiling labs

Compile and profile these separately:

- `launch_overhead.cu` — learn when tiny kernels become launch-bound.
- `bandwidth_copy.cu` — establish an effective memory-bandwidth baseline.
- `register_pressure.cu` — connect compiler register allocation to occupancy/spills.
- `gemm_naive_tiled.cu` — compare global-memory-heavy vs tiled GEMM.
- `fusion.cu` — compare launch/global-traffic savings against resource costs.

For every lab: benchmark → Nsight Systems → Nsight Compute (only dominant kernels) → hypothesis → one change → correctness → re-measure.

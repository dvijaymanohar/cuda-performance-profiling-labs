# Nsight workflow

1. Capture application/system behavior:
```bash
nsys profile -o profiles/timing ./build/timing_lab 50000000
```
2. Identify the dominant kernel or launch/transfer gap.
3. Profile only the relevant kernel:
```bash
ncu --set full -o profiles/kernel ./build/timing_lab 50000000
```
4. Inspect achieved bandwidth/compute, occupancy, register pressure/spills, warp stalls, memory transactions, and launch behavior.
5. State at least two plausible hypotheses before modifying code.

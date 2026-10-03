# Correct GPU benchmarking

- Define kernel-only vs end-to-end boundary.
- Warm up before steady-state measurement.
- Use CUDA events for device elapsed time.
- Synchronize correctly around asynchronous work.
- Repeat and report median plus spread.
- Keep input shape, dtype, hardware, clocks/power state, driver/runtime, and build flags recorded.
- Separate cold-start/setup from steady-state execution.
- Re-run correctness after every optimization.

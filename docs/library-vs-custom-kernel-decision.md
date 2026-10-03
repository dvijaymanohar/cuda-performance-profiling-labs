# Vendor library vs framework vs custom CUDA vs Triton

Do not assume a custom kernel is the best engineering choice.

## Decision questions
1. Is this operation materially on the end-to-end critical path?
2. Does the framework already generate/fuse an adequate kernel?
3. Is there a tuned vendor implementation (cuBLAS, cuBLASLt, cuDNN, CUTLASS-based path)?
4. Do shape, fusion, layout, or latency constraints require custom control?
5. Can Triton deliver the needed control with lower maintenance cost?
6. What portability, testing, upgrade, and team-maintenance costs does custom CUDA introduce?

## Required comparison
For one representative operation, compare applicable choices under identical shape, dtype, hardware, correctness tolerance, warm-up, and timing methodology. Record end-to-end impact as well as isolated kernel time.

The completion criterion includes at least one evidence-backed case where *not* writing custom CUDA is the correct decision.

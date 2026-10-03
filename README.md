# CUDA Performance & Profiling Labs

This repository turns a correct CUDA workload into an evidence-backed optimization study.

## Workflow
baseline → benchmark correctly → system timeline → dominant kernel → competing hypotheses → controlled change → re-benchmark → re-profile → correctness.

## Build
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/timing_lab 10000000
```

Do not use host wall-clock timing around asynchronous kernel launches without an explicit synchronization boundary.

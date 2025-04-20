# Benchmarking Instructions

For general reference see [BenchmarkTools.jl manual](https://juliaci.github.io/BenchmarkTools.jl/stable/manual/).

A simple way to run benchmarks is to call

```julia
using BenchmarkTools
using PkgBenchmark
using EquivariantTensors

bench = benchmarkpkg(EquivariantTensors)
results = bench.benchmarkgroup

# You can search with macro "@tagged"
results[@tagged "derivative" && "Chebyshev"]
```

You can create `BenchmarkConfig` to control benchmark

```julia
t2 = BenchmarkConfig(env = Dict("JULIA_NUM_THREADS" => 2))
bench_t2 = benchmarkpkg(EquivariantTensors, t2)
```

Benchmarks can be saved to a file with

```julia
export_markdown("results.md", bench)
```

Comparing current branch to another branch

```julia
# current branch to "origin/main"
j = judge(EquivariantTensors, "origin/main")
```

Benchmark scaling to different number of threads

```julia
t4 = BenchmarkConfig(env = Dict("JULIA_NUM_THREADS" => 4))
t8 = BenchmarkConfig(env = Dict("JULIA_NUM_THREADS" => 8))

# Compare how much changing from 4-threads to 8 improves the performance
j = judge(EquivariantTensors, t8, t4)

show(j.benchmarkgroup)
```

## CI Benchmarks

Benchmarks can be run automatically on PR's by adding label "run benchmark" to the PR.

## Adding more benchmarks

Take a look at `benchmark/benchmarks.jl` for an example. If your benchmark depends on additional packages you need to add the package to `benchmark/Project.toml`.

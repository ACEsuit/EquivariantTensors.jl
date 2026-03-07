using BenchmarkTools
SUITE = BenchmarkGroup()
include("bench_prodpool.jl")
include("bench_symmprod.jl")

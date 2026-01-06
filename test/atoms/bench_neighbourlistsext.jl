#
# Benchmark script for NeighbourLists integration
# Run with: julia --project test/atoms/bench_neighbourlistsext.jl
#
# This script compares the performance of the new multithreaded neighbour_list()
# API against the legacy PairList implementation.
#

using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using EquivariantTensors, NeighbourLists, AtomsBuilder, Unitful, BenchmarkTools
import EquivariantTensors as ET

# Benchmark configuration
system_sizes = [(5,5,5), (8,8,8), (10,10,10), (12,12,12)]
rcut = 5.0u"Å"

println("=" ^ 70)
println("NeighbourLists Integration Benchmarks")
println("=" ^ 70)
println("Cutoff: $rcut")
println("Julia threads: $(Threads.nthreads())")
println()

results = []

for mult in system_sizes
    sys = rattle!(bulk(:Si, cubic=true) * mult, 0.1u"Å")
    natoms = length(sys)

    println("\n" * "-" ^ 70)
    println("System: $mult supercell ($natoms atoms)")
    println("-" ^ 70)

    # Warm-up
    _ = ET.Atoms.interaction_graph(sys, rcut)
    _ = ET.Atoms.interaction_graph_legacy(sys, rcut)

    # Benchmark new API (materialized)
    t_new = @benchmark ET.Atoms.interaction_graph($sys, $rcut) samples=10 evals=1

    # Benchmark legacy API
    t_legacy = @benchmark ET.Atoms.interaction_graph_legacy($sys, $rcut) samples=10 evals=1

    # Benchmark lazy mode
    t_lazy = @benchmark ET.Atoms.interaction_graph($sys, $rcut; lazy=true) samples=10 evals=1

    speedup = median(t_legacy).time / median(t_new).time

    println("New API (materialized): $(round(median(t_new).time / 1e6, digits=2)) ms")
    println("Legacy API:             $(round(median(t_legacy).time / 1e6, digits=2)) ms")
    println("New API (lazy):         $(round(median(t_lazy).time / 1e6, digits=2)) ms")
    println("Speedup (new/legacy):   $(round(speedup, digits=2))x")

    push!(results, (mult=mult, natoms=natoms, t_new=median(t_new).time,
                    t_legacy=median(t_legacy).time, t_lazy=median(t_lazy).time,
                    speedup=speedup))
end

# Summary table
println("\n" * "=" ^ 70)
println("Summary")
println("=" ^ 70)
println()
println("| Supercell | Atoms | Legacy (ms) | New (ms) | Lazy (ms) | Speedup |")
println("|-----------|-------|-------------|----------|-----------|---------|")
for r in results
    println("| $(r.mult) | $(r.natoms) | $(round(r.t_legacy/1e6, digits=2)) | $(round(r.t_new/1e6, digits=2)) | $(round(r.t_lazy/1e6, digits=2)) | $(round(r.speedup, digits=2))x |")
end

# GPU benchmarks (if available)
println("\n" * "=" ^ 70)
println("GPU Transfer Benchmarks")
println("=" ^ 70)

include(joinpath(@__DIR__, "..", "test_utils", "utils_gpu.jl"))

if dev !== identity
    println("GPU device detected")
    println()

    for mult in system_sizes[1:min(3, length(system_sizes))]
        sys = rattle!(bulk(:Si, cubic=true) * mult, 0.1u"Å")
        G = ET.Atoms.interaction_graph(sys, rcut)

        # Warm-up
        _ = dev(G)

        t_transfer = @benchmark $dev($G) samples=10 evals=1

        println("$mult supercell GPU transfer: $(round(median(t_transfer).time / 1e6, digits=2)) ms")
    end
else
    println("No GPU available - skipping GPU benchmarks")
end

println("\n" * "=" ^ 70)
println("Benchmark complete")
println("=" ^ 70)

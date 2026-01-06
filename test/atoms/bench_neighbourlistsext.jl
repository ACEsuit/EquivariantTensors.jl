#
# Benchmark script for NeighbourLists integration
# Run with: julia --project=test test/atoms/bench_neighbourlistsext.jl
#
# This script compares the performance of the new multithreaded neighbour_list()
# API against the legacy PairList implementation.
#

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
println("GPU Benchmarks")
println("=" ^ 70)

include(joinpath(@__DIR__, "..", "test_utils", "utils_gpu.jl"))

using AtomsBase: position, cell_vectors, periodicity
using NeighbourLists: build_cell_list, materialize_pairlist, SVec
using StaticArrays: SVector

if dev !== identity
    println("GPU device detected")
    println()

    gpu_results = []

    for mult in system_sizes[1:min(3, length(system_sizes))]
        sys = rattle!(bulk(:Si, cubic=true) * mult, 0.1u"Å")
        natoms = length(sys)

        println("\n" * "-" ^ 70)
        println("GPU System: $mult supercell ($natoms atoms)")
        println("-" ^ 70)

        # Extract positions and cell for direct NeighbourLists calls
        pos = position(sys, :)
        X_cpu = [SVec(ustrip.(u"Å", p)...) for p in pos]
        rcut_val = ustrip(u"Å", rcut)

        # Get cell matrix
        cvecs = cell_vectors(sys)
        C = hcat([ustrip.(u"Å", c) for c in cvecs]...)
        pbc = periodicity(sys)

        # Convert positions to GPU
        X_gpu = dev(X_cpu)

        # Build on CPU (reference)
        clist_cpu = build_cell_list(X_cpu, rcut_val, C, pbc)
        _ = materialize_pairlist(clist_cpu)

        # Build on GPU (warm-up)
        clist_gpu = build_cell_list(X_gpu, rcut_val, C, pbc)
        _ = materialize_pairlist(clist_gpu)

        # Benchmark CPU build
        t_cpu_build = @benchmark materialize_pairlist(build_cell_list($X_cpu, $rcut_val, $C, $pbc)) samples=10 evals=1

        # Benchmark GPU build (positions already on GPU)
        t_gpu_build = @benchmark materialize_pairlist(build_cell_list($X_gpu, $rcut_val, $C, $pbc)) samples=10 evals=1

        # Benchmark CPU build + transfer to GPU
        G_cpu = ET.Atoms.interaction_graph(sys, rcut)
        _ = dev(G_cpu)
        t_cpu_then_transfer = @benchmark $dev(ET.Atoms.interaction_graph($sys, $rcut)) samples=10 evals=1

        gpu_speedup = median(t_cpu_build).time / median(t_gpu_build).time

        println("CPU build (NeighbourLists):  $(round(median(t_cpu_build).time / 1e6, digits=2)) ms")
        println("GPU build (NeighbourLists):  $(round(median(t_gpu_build).time / 1e6, digits=2)) ms")
        println("CPU build + GPU transfer:    $(round(median(t_cpu_then_transfer).time / 1e6, digits=2)) ms")
        println("GPU build speedup:           $(round(gpu_speedup, digits=2))x")

        push!(gpu_results, (mult=mult, natoms=natoms,
                           t_cpu=median(t_cpu_build).time,
                           t_gpu=median(t_gpu_build).time,
                           t_cpu_transfer=median(t_cpu_then_transfer).time,
                           speedup=gpu_speedup))
    end

    # GPU Summary table
    println("\n" * "=" ^ 70)
    println("GPU Summary")
    println("=" ^ 70)
    println()
    println("| Supercell | Atoms | CPU (ms) | GPU (ms) | CPU+Transfer (ms) | GPU Speedup |")
    println("|-----------|-------|----------|----------|-------------------|-------------|")
    for r in gpu_results
        println("| $(r.mult) | $(r.natoms) | $(round(r.t_cpu/1e6, digits=2)) | $(round(r.t_gpu/1e6, digits=2)) | $(round(r.t_cpu_transfer/1e6, digits=2)) | $(round(r.speedup, digits=2))x |")
    end
else
    println("No GPU available - skipping GPU benchmarks")
end

println("\n" * "=" ^ 70)
println("Benchmark complete")
println("=" ^ 70)

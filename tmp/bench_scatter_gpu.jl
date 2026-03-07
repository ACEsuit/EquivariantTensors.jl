#
# GPU Benchmark: fused KA vs scatter/gather vs fused-scatter
#
# Run:  julia --project=. tmp/bench_scatter_gpu.jl
#

using EquivariantTensors
using EquivariantTensors: PooledSparseProduct, evaluate,
   _generate_input, ka_evaluate, ka_evaluate!, ka_pullback,
   ka_pullback!
using CUDA
using KernelAbstractions

include(joinpath(@__DIR__, "scatter_prodpool.jl"))

# -------------------------------------------------------

function make_benchmark(; order = 2, nspec = 200,
                          nneig = 24, nnodes = 64)
   NN = [rand(20:40) for _ = 1:order]
   spec = sort([ntuple(t -> rand(1:NN[t]), order)
                for _ = 1:nspec])
   basis = PooledSparseProduct(spec)

   _bBB = _generate_input(basis; nX = nneig * nnodes)
   BB = ntuple(
      i -> Float32.(
         reshape(_bBB[i], (nneig, nnodes, :))), order)
   return basis, BB, nneig, nnodes
end

function bench_gpu(f; nwarmup = 10, nruns = 200)
   for _ in 1:nwarmup
      f()
   end
   CUDA.synchronize()
   times = Float64[]
   for _ in 1:nruns
      CUDA.synchronize()
      t = @elapsed begin
         f()
         CUDA.synchronize()
      end
      push!(times, t)
   end
   return minimum(times)
end

fmt(t) = rpad("$(round(t*1e6, digits=1))μs", 10)

# -------------------------------------------------------

println("="^70)
println("PooledSparseProduct GPU Benchmark (A100)")
println("  fused = existing KA kernel")
println("  sg    = scatter/gather (materialized intermediates)")
println("  fs    = fused-scatter (no intermediates)")
println("="^70)

# Verify correctness on GPU
@info("Verifying GPU correctness...")
basis, BB_cpu, nneig, nnodes = make_benchmark(;
   order = 3, nspec = 100, nneig = 16, nnodes = 32)
BB_gpu = cu.(BB_cpu)
spec_gpu = cu(basis.spec)

A_cpu = ka_evaluate(basis, BB_cpu, basis.spec,
                    nnodes, nneig)
∂A_cpu = randn(Float32, size(A_cpu))
∂A_gpu = cu(∂A_cpu)

∂BB_ka = ka_pullback(∂A_gpu, basis, BB_gpu,
                     spec_gpu, nnodes, nneig)
∂BB_sg = sg_pullback(∂A_gpu, basis, BB_gpu;
                     nneig, nnodes)
∂BB_fs = fs_pullback(∂A_gpu, basis, BB_gpu;
                     nneig, nnodes)
for t = 1:3
   e_sg = maximum(abs.(Array(∂BB_ka[t]) .-
                       Array(∂BB_sg[t])))
   e_fs = maximum(abs.(Array(∂BB_ka[t]) .-
                       Array(∂BB_fs[t])))
   println("  ∂BB[$t]: sg err=$e_sg  fs err=$e_fs")
end
println()

# -------------------------------------------------------

println("-"^70)
println(rpad("config", 38),
        rpad("fused", 10),
        rpad("sg", 10),
        rpad("fs", 10),
        "sg/fused  fs/fused")
println("-"^70)

for order in [2, 3, 4]
   for (nspec, nneig, nnodes) in [
         (100,  16,  64),
         (200,  24,  128),
         (500,  32,  256),
         (1000, 32,  512),
      ]

      basis, BB_cpu, nneig, nnodes = make_benchmark(;
         order, nspec, nneig, nnodes)
      BB_gpu = cu.(BB_cpu)
      spec_gpu = cu(basis.spec)
      gidx = GatherIndices(basis, BB_gpu)

      ∂A = CUDA.randn(Float32, nnodes, length(basis))

      # ----- Forward (fused vs sg only, fs uses fused) -----
      t_ka = bench_gpu() do
         ka_evaluate!(
            similar(BB_gpu[1], (nnodes, length(basis))),
            basis, BB_gpu, spec_gpu, nnodes, nneig)
      end
      t_sg = bench_gpu() do
         sg_evaluate!(
            similar(BB_gpu[1], (nnodes, length(basis))),
            basis, BB_gpu;
            gidx, nneig, nnodes)
      end

      cfg = "ord=$order n=$nspec ng=$nneig nd=$nnodes"
      println(rpad("fwd $cfg", 38),
              fmt(t_ka), fmt(t_sg), rpad("—", 10),
              rpad("$(round(t_sg/t_ka, digits=2))x", 10),
              "—")

      # ----- Backward (all three) -----
      t_ka_b = bench_gpu() do
         ka_pullback(∂A, basis, BB_gpu,
                     spec_gpu, nnodes, nneig)
      end
      t_sg_b = bench_gpu() do
         sg_pullback(∂A, basis, BB_gpu;
                     gidx, nneig, nnodes)
      end
      t_fs_b = bench_gpu() do
         fs_pullback(∂A, basis, BB_gpu;
                     gidx, nneig, nnodes)
      end

      println(rpad("bwd $cfg", 38),
              fmt(t_ka_b), fmt(t_sg_b), fmt(t_fs_b),
              rpad("$(round(t_sg_b/t_ka_b, digits=2))x",
                   10),
              "$(round(t_fs_b/t_ka_b, digits=2))x")
      println()
   end
end

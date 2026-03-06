#
# GPU Benchmark: scatter/gather vs fused KA prodpool
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

# -------------------------------------------------------

println("="^65)
println("PooledSparseProduct GPU Benchmark (A100): " *
        "fused KA vs scatter/gather")
println("="^65)

# First verify correctness on GPU
@info("Verifying GPU correctness...")
basis, BB_cpu, nneig, nnodes = make_benchmark(;
   order = 2, nspec = 100, nneig = 16, nnodes = 32)
BB_gpu = cu.(BB_cpu)
spec_gpu = cu(basis.spec)

A_ka = Array(ka_evaluate(basis, BB_gpu, spec_gpu,
                         nnodes, nneig))
A_sg = Array(sg_evaluate(basis, BB_gpu;
                         nneig, nnodes))
A_cpu = ka_evaluate(basis, BB_cpu, basis.spec,
                    nnodes, nneig)
println("  ka GPU vs CPU max error: ",
        maximum(abs.(A_ka .- A_cpu)))
println("  sg GPU vs CPU max error: ",
        maximum(abs.(A_sg .- A_cpu)))
println("  ka GPU vs sg GPU max error: ",
        maximum(abs.(A_ka .- A_sg)))

∂A_gpu = CUDA.randn(Float32, size(A_ka)...)
∂BB_ka = ka_pullback(∂A_gpu, basis, BB_gpu,
                     spec_gpu, nnodes, nneig)
∂BB_sg = sg_pullback(cu(Array(∂A_gpu)), basis, BB_gpu;
                     nneig, nnodes)
for t = 1:2
   err = maximum(abs.(Array(∂BB_ka[t]) .-
                      Array(∂BB_sg[t])))
   println("  pullback ∂BB[$t] max error: ", err)
end
println()

# -------------------------------------------------------

for order in [2, 3, 4]
   for (nspec, nneig, nnodes) in [
         (100, 16, 64),
         (200, 24, 128),
         (500, 32, 256),
         (1000, 32, 512),
      ]

      basis, BB_cpu, nneig, nnodes = make_benchmark(;
         order, nspec, nneig, nnodes)
      BB_gpu = cu.(BB_cpu)
      spec_gpu = cu(basis.spec)

      # Pre-allocate
      A_ka = ka_evaluate(basis, BB_gpu, spec_gpu,
                         nnodes, nneig)
      ∂A = CUDA.randn(Float32, size(A_ka)...)

      # Pre-compute gather indices on GPU
      gidx = GatherIndices(basis, BB_gpu)

      # Also pre-allocate for pullback
      ∂BB_ka = cu.(ntuple(
         t -> similar(BB_cpu[t]), order))

      println("--- order=$order  nspec=$nspec  " *
              "nneig=$nneig  nnodes=$nnodes ---")

      # Forward
      t_ka = bench_gpu() do
         ka_evaluate!(A_ka, basis, BB_gpu,
                      spec_gpu, nnodes, nneig)
      end
      A_sg = similar(A_ka)
      t_sg = bench_gpu() do
         sg_evaluate!(A_sg, basis, BB_gpu;
                      gidx, nneig, nnodes)
      end
      ratio_fwd = round(t_sg / t_ka, digits = 2)
      println("  fwd: fused=$(round(t_ka*1e6, digits=1))μs" *
              "  sg=$(round(t_sg*1e6, digits=1))μs" *
              "  ratio=$(ratio_fwd)x")

      # Backward
      t_ka_bwd = bench_gpu() do
         ka_pullback(∂A, basis, BB_gpu,
                     spec_gpu, nnodes, nneig)
      end
      t_sg_bwd = bench_gpu() do
         sg_pullback(∂A, basis, BB_gpu;
                     gidx, nneig, nnodes)
      end
      ratio_bwd = round(t_sg_bwd / t_ka_bwd, digits = 2)
      println("  bwd: fused=$(round(t_ka_bwd*1e6, digits=1))μs" *
              "  sg=$(round(t_sg_bwd*1e6, digits=1))μs" *
              "  ratio=$(ratio_bwd)x")
      println()
   end
end

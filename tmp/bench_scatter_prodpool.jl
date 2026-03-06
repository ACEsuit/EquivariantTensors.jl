#
# Benchmark: scatter/gather vs fused KA prodpool
#
# Run:  julia --project=. tmp/bench_scatter_prodpool.jl
#
# This benchmarks CPU performance. For GPU, load a GPU backend
# and convert arrays with `gpu(...)` before timing.
#

using EquivariantTensors
using EquivariantTensors: PooledSparseProduct, evaluate,
   _generate_input, ka_evaluate, ka_evaluate!, ka_pullback

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
      i -> reshape(_bBB[i], (nneig, nnodes, :)), order)

   return basis, BB, nneig, nnodes
end

function bench(f, nruns = 200)
   # warmup
   f()
   f()
   # timed runs
   t = minimum(@elapsed(f()) for _ in 1:nruns)
   return t
end

# -------------------------------------------------------

println("="^60)
println("PooledSparseProduct: fused KA vs scatter/gather (CPU)")
println("="^60)

for order in [2, 3, 4]
   for (nspec, nneig, nnodes) in [
         (100, 16, 32),
         (200, 24, 64),
         (500, 32, 128),
      ]

      basis, BB, nneig, nnodes = make_benchmark(;
         order, nspec, nneig, nnodes)
      gidx = GatherIndices(basis, BB)

      # pre-allocate outputs
      A_ka = ka_evaluate(basis, BB, basis.spec,
                         nnodes, nneig)
      A_sg = similar(A_ka)
      ∂A = randn(Float64, size(A_ka))

      println("\n--- order=$order  nspec=$nspec  " *
              "nneig=$nneig  nnodes=$nnodes ---")

      # Forward
      t_ka_fwd = bench() do
         ka_evaluate!(A_ka, basis, BB, basis.spec,
                      nnodes, nneig)
      end
      t_sg_fwd = bench() do
         sg_evaluate!(A_sg, basis, BB;
                      gidx, nneig, nnodes)
      end
      println("  fwd: fused=$(round(t_ka_fwd*1e6, digits=1))μs" *
              "  sg=$(round(t_sg_fwd*1e6, digits=1))μs" *
              "  ratio=$(round(t_sg_fwd/t_ka_fwd, digits=2))x")

      # Backward
      t_ka_bwd = bench() do
         ka_pullback(∂A, basis, BB, basis.spec,
                     nnodes, nneig)
      end
      t_sg_bwd = bench() do
         sg_pullback(∂A, basis, BB;
                     gidx, nneig, nnodes)
      end
      println("  bwd: fused=$(round(t_ka_bwd*1e6, digits=1))μs" *
              "  sg=$(round(t_sg_bwd*1e6, digits=1))μs" *
              "  ratio=$(round(t_sg_bwd/t_ka_bwd, digits=2))x")
   end
end

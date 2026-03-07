using EquivariantTensors
using EquivariantTensors: PooledSparseProduct, evaluate, pullback,
   _generate_input, ka_evaluate, ka_evaluate!, ka_pullback, ka_pullback!
using Random, LuxCore

# ============================================================
#  PooledSparseProduct benchmarks
# ============================================================

function _make_basis(; order = 3, nspec = 200)
   NN = [rand(20:40) for _ = 1:order]
   spec = sort([ntuple(t -> rand(1:NN[t]), order) for _ = 1:nspec])
   return PooledSparseProduct(spec)
end

function _make_ten3_input(basis::PooledSparseProduct{NB};
                          nneig = 24, nnodes = 64) where {NB}
   bBB = _generate_input(basis; nX = nneig * nnodes)
   BB = ntuple(i -> reshape(bBB[i], (nneig, nnodes, :)), NB)
   return BB, nneig, nnodes
end

SUITE["ProdPool"] = BenchmarkGroup()

# --- CPU: pooled (2D) ---
SUITE["ProdPool"]["cpu"] = BenchmarkGroup()

for order in [2, 3]
   basis = _make_basis(; order, nspec = 200)
   bBB = _generate_input(basis; nX = 64)
   SUITE["ProdPool"]["cpu"]["fwd ord=$order nX=64"] =
      @benchmarkable evaluate($basis, $bBB)
end

# --- KA: batched 3D (forward + backward) ---
SUITE["ProdPool"]["ka"] = BenchmarkGroup()

for (order, nspec, nneig, nnodes) in [
      (2, 200, 24, 64),
      (3, 200, 24, 64),
      (3, 500, 32, 128),
   ]
   basis = _make_basis(; order, nspec)
   BB, nneig, nnodes = _make_ten3_input(basis;
                                         nneig, nnodes)
   spec = basis.spec
   A = ka_evaluate(basis, BB, spec, nnodes, nneig)
   ∂A = randn(eltype(A), size(A))

   tag = "ord=$order n=$nspec ng=$nneig nd=$nnodes"
   SUITE["ProdPool"]["ka"]["fwd $tag"] =
      @benchmarkable ka_evaluate($basis, $BB, $spec,
                                 $nnodes, $nneig)
   SUITE["ProdPool"]["ka"]["bwd $tag"] =
      @benchmarkable ka_pullback($∂A, $basis, $BB,
                                 $spec, $nnodes, $nneig)
end

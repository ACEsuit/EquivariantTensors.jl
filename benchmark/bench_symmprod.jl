using EquivariantTensors
using EquivariantTensors: SparseSymmProd, evaluate, pullback,
   ka_evaluate, ka_pullback
using EquivariantTensors.Test_ACE: generate_SO2_spec

SUITE["SymmProd"] = BenchmarkGroup()
SUITE["SymmProd"]["ka"] = BenchmarkGroup()

for (ORD, M, nX) in [(2, 30, 64), (3, 25, 64), (3, 25, 128)]
   spec = generate_SO2_spec(ORD, 2*M)
   basis = SparseSymmProd(spec)
   A = randn(Float32, nX, 2*M+1)
   AA = ka_evaluate(basis, A, basis.specs)
   ∂AA = randn(Float32, size(AA))

   tag = "ord=$ORD M=$M nX=$nX"
   SUITE["SymmProd"]["ka"]["fwd $tag"] =
      @benchmarkable ka_evaluate($basis, $A, $(basis.specs))
   SUITE["SymmProd"]["ka"]["bwd $tag"] =
      @benchmarkable ka_pullback($∂AA, $basis, $A,
                                 $(basis.specs), $nX)
end

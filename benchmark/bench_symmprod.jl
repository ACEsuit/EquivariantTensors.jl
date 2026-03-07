using EquivariantTensors
using EquivariantTensors: SparseSymmProd, evaluate, pullback,
   ka_evaluate, ka_pullback
using LinearAlgebra: norm

function _generate_SO2_spec(order, M, p=1)
   i2m(i) = (-1)^(isodd(i-1)) * (i ÷ 2)
   spec = Vector{Int}[]
   function append_N!(::Val{N}) where {N}
      for ci in CartesianIndices(ntuple(_ -> 1:2*M+1, N))
         mm = i2m.(ci.I)
         if (sum(mm) == 0) && (norm(mm, p) <= M) && issorted(ci.I)
            push!(spec, [ci.I...,])
         end
      end
   end
   for N = 1:order
      append_N!(Val(N))
   end
   return spec
end

SUITE["SymmProd"] = BenchmarkGroup()
SUITE["SymmProd"]["ka"] = BenchmarkGroup()

for (ORD, M, nX) in [(2, 40, 1024), (3, 40, 1024), (4, 40, 1024)]
   spec = _generate_SO2_spec(ORD, 2*M)
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

# Tests for the row-wise pushforwards `pushforward_rows` / `pushforward_rows!` 
# of PooledSparseProduct, SparseSymmProd and SparseACEbasis: one tangent 
# per input row (neighbour), with scalar or SVector{3} tangent types.

using Test, EquivariantTensors, StaticArrays, LinearAlgebra, Random
using EquivariantTensors: PooledSparseProduct, SparseSymmProd, evaluate, 
         _generate_input
using ACEbase.Testing: fdtest, println_slim, print_tf 
import EquivariantTensors as ET 
import ForwardDiff
import Polynomials4ML as P4ML 
import Lux

isdefined(Main, :__TEST_ACE__) || include("utils_ace.jl")

##

_rows_dot(∂X::AbstractMatrix, U::AbstractVector) = 
      [ sum(dot(∂X[j, k], U[j]) for j = 1:size(∂X, 1)) for k = 1:size(∂X, 2) ]

# contract per-row tangents ∂X[j, k] with directions U[j] -> scalar tangents 
_scalar_tangent(∂X::AbstractMatrix, U::AbstractVector) = 
      [ dot(∂X[j, k], U[j]) for j = 1:size(∂X, 1), k = 1:size(∂X, 2) ]

# ⟨∂Y, W⟩ summed over rows, ∂Y[j, k] scalar or SVector 
_contract(W::AbstractVector, ∂Y::AbstractMatrix) = 
      sum(W[k] * ∂Y[j, k] for j = 1:size(∂Y, 1), k = 1:size(∂Y, 2))

# allocations of the in-place call itself, behind a function barrier with 
# explicit arguments and the returned tuple discarded (a varargs wrapper or 
# a call at global scope boxes the tuple, which is not part of the kernel) 
_pf_rows_discard!(Y, ∂Y, basis, X, ∂X) = 
      (ET.pushforward_rows!(Y, ∂Y, basis, X, ∂X); nothing)
_pf_rows_discard!(B, ∂B, tensor, Rnl, Ylm, ∂Rnl, ∂Ylm) = 
      (ET.pushforward_rows!(B, ∂B, tensor, Rnl, Ylm, ∂Rnl, ∂Ylm); nothing)
_nalloc(Y, ∂Y, basis, X, ∂X) = @allocated _pf_rows_discard!(Y, ∂Y, basis, X, ∂X)
_nalloc(B, ∂B, tensor, Rnl, Ylm, ∂Rnl, ∂Ylm) = 
      @allocated _pf_rows_discard!(B, ∂B, tensor, Rnl, Ylm, ∂Rnl, ∂Ylm)

_rand_tangent(::Type{T}, ::Val{:svec}, dims...) where {T} = 
      randn(SVector{3, T}, dims...)
_rand_tangent(::Type{T}, ::Val{:scalar}, dims...) where {T} = 
      randn(T, dims...)

function _generate_pooled_basis(::Type{T}; order = 2, len = 40) where {T}
   NN = [ rand(8:15) for _ = 1:order ]
   spec = sort(unique([ ntuple(t -> rand(1:NN[t]), order) for _ = 1:len ]))
   return PooledSparseProduct(spec), NN
end

# an AA spec over nA A-functions with all orders 1..ORD (random subset)
function _generate_aa_spec(nA, ORD; hasconst = false, len = 40)
   spec = Vector{Int}[] 
   hasconst && push!(spec, Int[])
   for N = 1:ORD
      cands = unique([ sort([ rand(1:nA) for _ = 1:N ]) for _ = 1:len ])
      append!(spec, cands)
   end
   return spec 
end

##

@info("Row-wise pushforward of PooledSparseProduct")

for T in (Float64, Float32), TT in (:svec, :scalar), NB in 1:4 
   local rtol = (T == Float64) ? 1e-10 : 1e-4
   basis, NN = _generate_pooled_basis(T; order = NB)
   nX = rand(5:12)
   BB = ntuple(t -> randn(T, nX, NN[t]), NB)
   ∂BB = ntuple(t -> _rand_tangent(T, Val(TT), nX, NN[t]), NB)

   A, ∂A = ET.pushforward_rows(basis, BB, ∂BB)
   print_tf(@test eltype(A) == T && eltype(∂A) == eltype(∂BB[1]))
   print_tf(@test size(∂A) == (nX, length(basis)))
   print_tf(@test A ≈ evaluate(basis, BB))

   # (a) contract with a random direction per row -> scalar tangent problem, 
   #     compare with ForwardDiff and with the existing (Dual) pushforward 
   U = (TT == :svec) ? randn(SVector{3, T}, nX) : ones(T, nX)
   ∂BBs = ntuple(t -> _scalar_tangent(∂BB[t], U), NB)
   _BB(h) = ntuple(t -> BB[t] + h * ∂BBs[t], NB)
   dA_fd = ForwardDiff.derivative(h -> evaluate(basis, _BB(h)), zero(T))
   dA_rows = _rows_dot(∂A, U)
   print_tf(@test isapprox(dA_rows, dA_fd; rtol = rtol))
   _, ∂A_dual = ET.pushforward(basis, BB, ∂BBs)
   print_tf(@test isapprox(dA_rows, ∂A_dual; rtol = rtol))

   # (b) finite differences 
   if T == Float64 
      W = randn(length(basis)) ./ (1:length(basis))
      F(h) = dot(W, evaluate(basis, _BB(h)))
      dF(h) = dot(W, vec(sum(ET.pushforward_rows(basis, _BB(h), ∂BBs)[2], dims = 1)))
      print_tf(@test fdtest(F, dF, 0.0; verbose = false))
   end

   # (c) adjoint identity with the pullback: ⟨W, ∂A⟩ = ⟨pullback(W), ∂BB⟩
   W = randn(T, length(basis))
   lhs = _contract(W, ∂A)
   ∂BB_pb = ET.pullback(W, basis, BB)
   rhs = sum(_contract(vec(∂BB_pb[t]), reshape(∂BB[t], 1, :)) for t = 1:NB)
   print_tf(@test isapprox(lhs, rhs; rtol = rtol))

   # type stability and allocations of the in-place version 
   A1, ∂A1 = copy(A), copy(∂A)
   print_tf(@test (@inferred ET.pushforward_rows!(A1, ∂A1, basis, BB, ∂BB)) isa 
                  Tuple{typeof(A1), typeof(∂A1)})
   print_tf(@test A1 == A && ∂A1 == ∂A)
   print_tf(@test _nalloc(A1, ∂A1, basis, BB, ∂BB) == 0)
end
println()

##

@info("Row-wise pushforward of SparseSymmProd")

for T in (Float64, Float32), TT in (:svec, :scalar), ORD in 2:4, hasconst in (false, true)
   local rtol = (T == Float64) ? 1e-10 : 1e-4
   # A and ∂A come from a PooledSparseProduct{2} row-wise pushforward 
   abasis, NN = _generate_pooled_basis(T; order = 2, len = 30)
   nX = rand(5:12)
   BB = ntuple(t -> randn(T, nX, NN[t]), 2)
   ∂BB = ntuple(t -> _rand_tangent(T, Val(TT), nX, NN[t]), 2)
   A, ∂A = ET.pushforward_rows(abasis, BB, ∂BB)
   nA = length(abasis)

   basis = SparseSymmProd(_generate_aa_spec(nA, ORD; hasconst = hasconst))
   print_tf(@test basis.hasconst == hasconst)
   AA, ∂AA = ET.pushforward_rows(basis, A, ∂A)
   print_tf(@test eltype(AA) == T && eltype(∂AA) == eltype(∂A))
   print_tf(@test size(∂AA) == (nX, length(basis)))
   print_tf(@test AA ≈ evaluate(basis, A))
   if hasconst 
      print_tf(@test AA[1] == 1 && all(iszero, ∂AA[:, 1]))
   end

   # (a) ForwardDiff and the existing Dual pushforward on the scalar-tangent problem 
   U = (TT == :svec) ? randn(SVector{3, T}, nX) : ones(T, nX)
   ∂BBs = ntuple(t -> _scalar_tangent(∂BB[t], U), 2)
   _BB(h) = ntuple(t -> BB[t] + h * ∂BBs[t], 2)
   _A(h) = evaluate(abasis, _BB(h))
   dAA_fd = ForwardDiff.derivative(h -> evaluate(basis, _A(h)), zero(T))
   dAA_rows = _rows_dot(∂AA, U)
   print_tf(@test isapprox(dAA_rows, dAA_fd; rtol = rtol))
   _, ∂AA_dual = ET.pushforward(basis, A, _rows_dot(∂A, U))
   print_tf(@test isapprox(dAA_rows, ∂AA_dual; rtol = rtol))

   # (b) finite differences 
   if T == Float64
      W = randn(length(basis)) ./ (1:length(basis))
      F(h) = dot(W, evaluate(basis, _A(h)))
      dF(h) = begin 
         Ah, ∂Ah = ET.pushforward_rows(abasis, _BB(h), ∂BBs)
         dot(W, vec(sum(ET.pushforward_rows(basis, Ah, ∂Ah)[2], dims = 1)))
      end
      print_tf(@test fdtest(F, dF, 0.0; verbose = false))
   end

   # (c) adjoint identity with the pullback: ⟨W, ∂AA⟩ = ⟨pullback(W), ∂A⟩
   W = randn(T, length(basis))
   lhs = _contract(W, ∂AA)
   rhs = _contract(ET.pullback(W, basis, A), ∂A)
   print_tf(@test isapprox(lhs, rhs; rtol = rtol))

   # type stability and allocations of the in-place version 
   AA1, ∂AA1 = copy(AA), copy(∂AA)
   print_tf(@test (@inferred ET.pushforward_rows!(AA1, ∂AA1, basis, A, ∂A)) isa 
                  Tuple{typeof(AA1), typeof(∂AA1)})
   print_tf(@test AA1 == AA && ∂AA1 == ∂AA)
   print_tf(@test _nalloc(AA1, ∂AA1, basis, A, ∂A) == 0)
end
println()

##

@info("Row-wise pushforward of SparseACEbasis (L = 0)")

# a small invariant (L = 0) ACE basis; Rnl, Ylm and their position 
# derivatives come from P4ML so that ∂B is the Jacobian w.r.t. positions 
Dtot = 6; maxl = 3; ORD = 3
rbasis = P4ML.legendre_basis(Dtot+1)
ybasis = P4ML.real_sphericalharmonics(maxl)
nnll = ET.sparse_nnll_set(; ORD = ORD, minn = 0, maxn = Dtot, maxl = maxl, 
                            level = bb -> sum((b.n + b.l) for b in bb; init = 0), 
                            maxlevel = Dtot)
tensor = ET.sparse_equivariant_tensors(; LL = (0,), mb_spec = nnll, 
                                         Rnl_spec = P4ML.natural_indices(rbasis), 
                                         Ylm_spec = P4ML.natural_indices(ybasis), 
                                         basis = real)
print_tf(@test length(tensor.A2Bmaps) == 1)

__rand_sphere() = ( u = randn(SVector{3, Float64}); u / norm(u) )
__rand_x() = (0.1 + 0.8 * rand()) * __rand_sphere()

function _embed(𝐫::AbstractVector{SVector{3, T}}) where {T}
   rs = norm.(𝐫)
   Rnl = P4ML.evaluate(rbasis, rs)
   Ylm = P4ML.evaluate(ybasis, 𝐫)
   return Rnl, Ylm 
end

function _embed_ed(𝐫::AbstractVector{SVector{3, T}}) where {T}
   rs = norm.(𝐫)
   Rnl, dRnl = P4ML.evaluate_ed(rbasis, rs)
   ∂Rnl = [ dRnl[j, n] * 𝐫[j] / rs[j] for j = 1:length(𝐫), n = 1:size(Rnl, 2) ]
   Ylm, ∂Ylm = P4ML.evaluate_ed(ybasis, 𝐫)
   return Rnl, Ylm, ∂Rnl, ∂Ylm 
end

_B(𝐫) = evaluate(tensor, _embed(𝐫)...)[1]

for ntest = 1:10 
   local nX, 𝐫, Rnl, Ylm, ∂Rnl, ∂Ylm, B, ∂B, U, W, lhs, rhs, A, B1, ∂B1
   nX = rand(4:10)
   𝐫 = [ __rand_x() for _ = 1:nX ]
   Rnl, Ylm, ∂Rnl, ∂Ylm = _embed_ed(𝐫)
   B, ∂B = ET.pushforward_rows(tensor, Rnl, Ylm, ∂Rnl, ∂Ylm)
   print_tf(@test B isa Vector{Float64} && ∂B isa Matrix{SVector{3, Float64}})
   print_tf(@test size(∂B) == (nX, length(tensor)))
   print_tf(@test B ≈ _B(𝐫))

   # (a) directional derivative w.r.t. positions via ForwardDiff 
   U = randn(SVector{3, Float64}, nX)
   dB_fd = ForwardDiff.derivative(h -> _B(𝐫 + h * U), 0.0)
   print_tf(@test isapprox(_rows_dot(∂B, U), dB_fd; rtol = 1e-10))

   # (b) finite differences 
   W = randn(length(tensor)) ./ (1:length(tensor))
   F(h) = dot(W, _B(𝐫 + h * U))
   dF(h) = dot(W, _rows_dot(ET.pushforward_rows(tensor, _embed_ed(𝐫 + h * U)...)[2], U))
   print_tf(@test fdtest(F, dF, 0.0; verbose = false))

   # (c) adjoint identity with the tensor pullback 
   A = evaluate(tensor.abasis, (Rnl, Ylm))
   ∂Rnl_pb, ∂Ylm_pb = ET.pullback([W], tensor, Rnl, Ylm, A)
   lhs = _contract(W, ∂B)
   rhs = sum(∂Rnl_pb .* ∂Rnl) + sum(∂Ylm_pb .* ∂Ylm)
   print_tf(@test isapprox(lhs, rhs; rtol = 1e-10))

   # in-place: type stability, allocations (intermediates are Bumper-allocated) 
   B1, ∂B1 = copy(B), copy(∂B)
   print_tf(@test (@inferred ET.pushforward_rows!(B1, ∂B1, tensor, Rnl, Ylm, ∂Rnl, ∂Ylm)) isa 
                  Tuple{typeof(B1), typeof(∂B1)})
   print_tf(@test B1 == B && ∂B1 == ∂B)
   print_tf(@test _nalloc(B1, ∂B1, tensor, Rnl, Ylm, ∂Rnl, ∂Ylm) < 1_000)
end
println()

##

@info("SparseACEbasis: dense coupling matrix fallback and Float32")

let 
   nX = 7
   𝐫 = [ __rand_x() for _ = 1:nX ]
   Rnl, Ylm, ∂Rnl, ∂Ylm = _embed_ed(𝐫)
   B, ∂B = ET.pushforward_rows(tensor, Rnl, Ylm, ∂Rnl, ∂Ylm)
   # dense fallback of the coupling step 
   A, ∂A = ET.pushforward_rows(tensor.abasis, (Rnl, Ylm), (∂Rnl, ∂Ylm))
   AA, ∂AA = ET.pushforward_rows(tensor.aabasis, A, ∂A)
   C = Matrix(tensor.A2Bmaps[1])
   Bd = zeros(1, length(tensor)); ∂Bd = zeros(SVector{3, Float64}, nX, length(tensor))
   ET._mul_A2Bt!(Bd, reshape(AA, 1, :), C)
   ET._mul_A2Bt!(∂Bd, ∂AA, C)
   println_slim(@test vec(Bd) ≈ B && ∂Bd ≈ ∂B)
   # Float32 inputs; the output promotes with the coupling coefficients 
   B32, ∂B32 = ET.pushforward_rows(tensor, Float32.(Rnl), Float32.(Ylm), 
                                   SVector{3, Float32}.(∂Rnl), SVector{3, Float32}.(∂Ylm))
   TB = promote_type(Float32, eltype(tensor.A2Bmaps[1]))
   println_slim(@test B32 isa Vector{TB} && ∂B32 isa Matrix{SVector{3, TB}})
   println_slim(@test isapprox(B32, B; rtol = 1e-4))
   println_slim(@test isapprox(reinterpret(TB, ∂B32), reinterpret(Float64, ∂B); rtol = 1e-3))
end

##

@info("Batched _jacobian_X agrees with pushforward_rows node by node")

# _jacobian_X is the batched (maxneigs x nnodes x nfeat) form of 
# pushforward_rows; on the CPU both run the same row-wise kernels. 

_node(X::AbstractArray{<: Any, 3}, i) = X[:, i, :]

for T in (Float64, Float32), TT in (:svec, :scalar), NB in 1:4 
   local basis, NN = _generate_pooled_basis(T; order = NB)
   local maxneigs, nnodes = rand(5:12), rand(2:4)
   local BB = ntuple(t -> randn(T, maxneigs, nnodes, NN[t]), NB)
   local ∂BB = ntuple(t -> _rand_tangent(T, Val(TT), maxneigs, nnodes, NN[t]), NB)
   local A, ∂A = ET._jacobian_X(basis, BB, ∂BB)
   print_tf(@test eltype(∂A) == eltype(∂BB[1]))
   print_tf(@test size(A) == (nnodes, length(basis)) && 
                  size(∂A) == (maxneigs, nnodes, length(basis)))
   for i = 1:nnodes 
      Ai, ∂Ai = ET.pushforward_rows(basis, ntuple(t -> _node(BB[t], i), NB), 
                                           ntuple(t -> _node(∂BB[t], i), NB))
      print_tf(@test A[i, :] ≈ Ai && _node(∂A, i) ≈ ∂Ai)
   end
end
println()

for T in (Float64, Float32), TT in (:svec, :scalar), ORD in 2:4, hasconst in (false, true)
   local nA, nnodes, maxneigs = 12, rand(2:4), rand(5:12)
   local basis = SparseSymmProd(_generate_aa_spec(nA, ORD; hasconst = hasconst))
   local A = randn(T, nnodes, nA)
   local ∂A = _rand_tangent(T, Val(TT), maxneigs, nnodes, nA)
   local AA, ∂AA = ET._jacobian_X(basis, A, ∂A, basis.specs)
   for i = 1:nnodes 
      AAi, ∂AAi = ET.pushforward_rows(basis, A[i, :], _node(∂A, i))
      print_tf(@test AA[i, :] ≈ AAi && _node(∂AA, i) ≈ ∂AAi)
   end
   # the constant term is 1 with zero derivative 
   if hasconst 
      print_tf(@test all(AA[:, 1] .== 1) && all(iszero, ∂AA[:, :, 1]))
   end
end
println()

# full tensor, batched over nodes, with SVector{3} tangents 
let nnodes = 3, maxneigs = 7 
   local 𝐫s = [ [ __rand_x() for _ = 1:maxneigs ] for _ = 1:nnodes ]
   local emb = _embed_ed.(𝐫s)
   _stack(k) = permutedims(cat([ e[k] for e in emb ]...; dims = 3), (1, 3, 2))
   local Rnl, Ylm, ∂Rnl, ∂Ylm = _stack(1), _stack(2), _stack(3), _stack(4)
   print_tf(@test size(Rnl) == (maxneigs, nnodes, length(rbasis)) && 
                  eltype(∂Rnl) == SVector{3, Float64})
   local st = Lux.initialstates(Random.default_rng(), tensor)
   local (𝔹,), (∂𝔹,) = ET._jacobian_X(tensor, Rnl, Ylm, ∂Rnl, ∂Ylm, NamedTuple(), st)
   print_tf(@test eltype(∂𝔹) == SVector{3, Float64})
   for i = 1:nnodes 
      Bi, ∂Bi = ET.pushforward_rows(tensor, emb[i]...)
      print_tf(@test 𝔹[i, :] ≈ Bi && _node(∂𝔹, i) ≈ ∂Bi)
   end
end
println()

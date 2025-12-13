
using Test, EquivariantTensors, ChainRulesCore
using EquivariantTensors: PooledSparseProduct, evaluate, evaluate!, 
         _generate_input, _generate_input_1
# using Polynomials4ML.Testing: test_withalloc
using ACEbase.Testing: fdtest, println_slim, print_tf 

import EquivariantTensors as ET 

test_evaluate(basis::PooledSparseProduct, BB::Tuple{Vararg{AbstractVector}}) =
   [prod(BB[j][basis.spec[i][j]] for j = 1:length(BB))
    for i = 1:length(basis)]

test_evaluate(basis::PooledSparseProduct, BB::Tuple{Vararg{AbstractMatrix}}) =
   sum(test_evaluate(basis, ntuple(i -> BB[i][j, :], length(BB)))
       for j = 1:size(BB[1], 1))


##

function _generate_basis(; order=3, len = 50)
   NN = [ rand(10:30) for _ = 1:order ]
   spec = sort([ ntuple(t -> rand(1:NN[t]), order) for _ = 1:len])
   return PooledSparseProduct(spec)
end


##

@info("Test evaluation with a single input (no pooling)")

for ntest = 1:30
   local BB, A1, A2, basis

   order = mod1(ntest, 4)
   basis = _generate_basis(; order=order)
   BB = _generate_input_1(basis)
   A1 = test_evaluate(basis, BB)
   A2 = evaluate(basis, BB)
   print_tf(@test A1 ≈ A2)
end
println()

## 

@info("Test pooling of multiple inputs")
nX = 17

for ntest = 1:30 
   local bBB, bA1, bA2, bA3, basis 

   order = mod1(ntest, 4)
   basis = _generate_basis(; order=order)
   bBB = _generate_input(basis)
   bA1 = test_evaluate(basis, bBB)
   bA2 = evaluate(basis, bBB)
   bA3 = copy(bA2)
   evaluate!(bA3, basis, bBB)
   print_tf(@test bA1 ≈ bA2 ≈ bA3)
end

println()

##

# TODO: revive test to check no allocations 
# @info("    testing withalloc")
# basis = _generate_basis(; order=2)
# BB = _generate_input_1(basis)
# bBB = _generate_input(basis)
# test_withalloc(basis; batch=false)

##

@info("Testing rrule")
using LinearAlgebra: dot

for ntest = 1:30
   local bBB, bA2, u, basis, nX 
   order = mod1(ntest, 4)
   basis = _generate_basis(; order=order)
   bBB = _generate_input(basis)
   nX = size(bBB[1], 1)
   bUU = _generate_input(basis; nX = nX) # same shape and type as bBB 
   _BB(t) = ntuple(i -> bBB[i] + t * bUU[i], order)
   bA2 = evaluate(basis, bBB)
   u = randn(size(bA2))
   F(t) = dot(u, evaluate(basis, _BB(t)))
   dF(t) = begin
      val, pb = rrule(evaluate, basis, _BB(t))
      ∂BB = pb(u)[3]
      return sum(dot(∂BB[i], bUU[i]) for i = 1:length(bUU))
   end
   print_tf(@test fdtest(F, dF, 0.0; verbose=false))
end
println()

## 

@info("Testing pullback2 for PooledSparseProduct")
import ChainRulesCore: rrule, NoTangent

for ntest = 1:20
   local basis, val, pb, bBB, A 
   ORDER = mod1(ntest, 4)
   basis = _generate_basis(;order = ORDER)
   bBB = _generate_input(basis)
   ∂A = randn(length(basis))

   A = evaluate(basis, bBB)
   val, pb = rrule(evaluate, basis, bBB)
   nt1, nt2, ∂_BB = pb(∂A)

   @test val ≈ A
   @test nt1 isa NoTangent && nt2 isa NoTangent
   @test ∂_BB isa NTuple{ORDER, <: AbstractMatrix}
   @test all(size(∂_BB[i]) == size(bBB[i]) for i = 1:length(bBB))

   val2, pb2 = rrule(ET.pullback, ∂A, basis, bBB)
   @test val2 == ∂_BB

   ∂2 = ntuple(i -> randn(size(∂_BB[i])), length(∂_BB))
   bUU = _generate_input(basis; nX = size(bBB[1], 1))
   _BB(t) = ntuple(i -> bBB[i] + t * bUU[i], ORDER)
   bV = randn(size(∂A))
   _∂A(t) = ∂A + t * bV

   F(t) = begin
      ∂_BB = ET.pullback(_∂A(t), basis, _BB(t))
      return sum(dot(∂2[i], ∂_BB[i]) for i = 1:length(∂_BB))
   end
   dF(t) = begin
      val, pb = rrule(ET.pullback, ∂A, basis, _BB(t))
      _, ∂_∂A, _, ∂2_BB = pb(∂2)
      return dot(∂_∂A, bV) + sum(dot(bUU[i], ∂2_BB[i]) for i = 1:ORDER)
   end

   print_tf(@test fdtest(F, dF, 0.0; verbose=false) )
end
println()


## 
@info("Testing pushforward for PooledSparseProduct")


for ntest = 1:20 
   local order, basis, BB, ΔBB, A1, ∂A1, A2, ∂A2, A
   order = mod1(ntest, 4)
   basis = _generate_basis(; order=order)
   BB = _generate_input(basis) 
   ΔBB = ntuple(i -> randn(Float64, size(BB[i])), order) 
   _BB(t) = ntuple(i -> BB[i] + t * ΔBB[i], order)
   U = randn(length(basis)) ./ (1:length(basis))
   A, ∂A = ET.pushforward(basis, BB, ΔBB)
   print_tf(@test A ≈ basis(BB))
   F(t) = dot(U, evaluate(basis, _BB(t)))
   dF(t) = dot(U, ET.pushforward(basis, _BB(t), ΔBB)[2])
   print_tf(@test fdtest(F, dF, 0.0; verbose=false))
end
println() 

## 

@info("Trying to load a GPU device") 
include(joinpath(@__DIR__(), "..", "test_utils", "utils_gpu.jl"))


@info("Testing KA implementation of PooledSparseProduct") 

ntest = 10
itest = 2

for itest = 1:ntest       
   local nX, basis, P1, P2  
   order = mod1(itest, 3)
   basis = _generate_basis(; order=order, len = rand(50:200))
   BB = _generate_input(basis) 
   BB_32 = ntuple(i -> Float32.(BB[i]), length(BB))
   BB_gpu = gpu.(BB_32)
   nX = size(BB[1], 1)

   # Float64 tests, CPU only  
   P1 = evaluate(basis, BB)
   P2 = similar(P1) 
   ET.ka_evaluate!(P2, basis, BB)
   print_tf(@test P1 ≈ P2)

   # Float32 tests, CPU and GPU 
   P3 = Float32.(P2)
   ET.ka_evaluate!(P3, basis, BB_32)
   P4 = gpu(similar(P3))
   ET.ka_evaluate!(P4, basis, BB_gpu, gpu(basis.spec), nX)
   print_tf(@test Float32.(P1) ≈ P3) 
   print_tf(@test Float32.(P1) ≈ Array(P4))   

   # test with batch of inputs 
   nneig = rand(16:32) 
   _bBB = _generate_input(basis; nX = nneig * nX)
   bBB = ntuple(i -> Float32.(collect(reshape(_bBB[i], (nneig, nX, :)))), order)
   bBB_gpu = gpu.(bBB)
   bP1 = evaluate(basis, bBB)
   bP2 = similar(bP1)
   ET.ka_evaluate!(bP2, basis, bBB, basis.spec)
   print_tf(@test bP1 ≈ bP2)
   bP3 = gpu(similar(bP2))
   ET.ka_evaluate!(bP3, basis, bBB_gpu, gpu(basis.spec))
   print_tf(@test Float32.(bP1) ≈ Array(bP3))

   # test pullback
   ∂P = randn(Float32, size(bP1))
   ∂BB1 = ET.pullback(∂P, basis, bBB)
   ∂BB2 = ET.ka_pullback(∂P, basis, bBB)
   ∂BB3 = ET.ka_pullback(gpu(∂P), basis, bBB_gpu, gpu(basis.spec), nX)
   all(∂BB1[t] ≈ ∂BB2[t] ≈ Array(∂BB3[t]) for t = 1:length(∂BB1)) 
end
println() 

##

# temporary test to evaluate with dual numbers as input 
# this needs to be revisited / extended / redesigned

using StaticArrays
import ForwardDiff as FD
_dualize(A::AbstractArray{<: Number}) = FD.Dual.(A, one(eltype(A)))
_dualize(x) = x
_dualize(nt::NamedTuple) = NamedTuple{keys(nt)}( _dualize.(values(nt)) )

order = 2
basis = _generate_basis(; order=order, len = rand(50:200))
BB = _generate_input(basis) 
BB_32 = ntuple(i -> Float32.(BB[i]), length(BB))
BB_gpu = gpu.(BB_32)
nX = size(BB[1], 1)
spec_gpu = gpu(basis.spec)

# Float64 tests, CPU only  
P1 = evaluate(basis, BB)
P2 = similar(P1) 
ET.ka_evaluate!(P2, basis, BB)
print_tf(@test P1 ≈ P2)

BB_d = _dualize.(BB)
P1_d = ET.evaluate(basis, BB_d)

# ------

P3 = Float32.(P2)
ET.ka_evaluate!(P3, basis, BB_32)

##

P4 = gpu(similar(P3))
ET.ka_evaluate!(P4, basis, BB_gpu, spec_gpu, nX)

BB_gpu_d = _dualize.(BB_gpu)
P4_d = _dualize(P4)
ET.ka_evaluate!(P4_d, basis, BB_gpu_d, spec_gpu, nX)



using Test, Random, EquivariantTensors
using EquivariantTensors: SparseSymmProd, evaluate
using ACEbase.Testing: println_slim, print_tf, fdtest, dirfdtest
using ChainRulesCore: rrule 

ET = EquivariantTensors

isdefined(Main, :__TEST_ACE__) || include("utils_ace.jl")

#TODO
                              # test_withalloc

##

M = 5 
spec = Test_ACE.generate_SO2_spec(5, 2*M)
A = randn(ComplexF64, 2*M+1)

## 

@info("Test consistency of SparseSymmetricProduct with SimpleProdBasis")
basis1 = Test_ACE.SimpleProdBasis(spec)
AA1 = basis1(A)

basis2 = SparseSymmProd(spec)
AA2 = basis2(A)

@info("check against simple implementation")
println_slim(@test AA1 ≈ AA2)

@info("reconstruct spec")
spec_ = ET.reconstruct_spec(basis2)
println_slim(@test spec_ == spec)

println_slim(@test basis2.hasconst == false)

##

@info("Test with a constant")
spec_c = [ [Int[],]; spec]
basis1_c = Test_ACE.SimpleProdBasis(spec_c)
basis2_c = SparseSymmProd(spec_c)

println_slim(@test basis2_c.hasconst == true)

spec_c_ = ET.reconstruct_spec(basis2_c)
println_slim(@test spec_c_ == spec_c)

AA1_c = basis1_c(A)
println_slim(@test AA1 ≈ AA1_c[2:end])
println_slim(@test AA1_c[1] ≈ 1.0)

AA2_c = basis2_c(A)
println_slim(@test AA2_c[1] ≈ 1.0)
println_slim(@test AA2_c ≈ AA1_c)


## 

@info("Test gradient of SparseSymmetricProduct") 

using LinearAlgebra: dot
using Printf

for ntest = 1:10 
   local A, AA, Δ, f, g, pb, g0, dg0, errs, h, δA

   A = randn(2*M+1)
   AA = basis2(A)
   Δ = randn(length(AA)) ./ (1+length(AA))

   f(A) = dot(basis2(A), Δ)
   f(A)

   δA = randn(length(A)) ./ (1+length(A))
   g(t) = f(A + t * δA)

   AA, pb = rrule(ET.evaluate, basis2, A)
   g0 = dot(Δ, AA)
   dg0 = dot(pb(Δ)[3], δA)

   errs = Float64[]
   for h = (0.1).^(0:10)
      push!(errs, abs((g(h) - g0)/h - dg0))
      # @printf(" %.2e | %.2e \n", h, errs[end])
   end
   /(extrema(errs)...)
   print_tf(@test /(extrema(errs)...) < 1e-4)
end
println() 

## 

@info("Test consistency of serial and batched evaluation")

nX = 32
bA = randn(ComplexF64, nX, 2*M+1)
bAA1 = zeros(ComplexF64, nX, length(spec))
for i = 1:nX
   bAA1[i, :] = basis1(bA[i, :])
end
bAA2 = basis2(bA)

println_slim(@test bAA1 ≈ bAA2)

## 

@info("Test consistency of serial and batched evaluation with constant")

nX = 32
bA = randn(ComplexF64, nX, 2*M+1)
bAA1 = zeros(ComplexF64, nX, length(basis1_c))
for i = 1:nX
   bAA1[i, :] = basis1_c(bA[i, :])
end
bAA2 = basis2_c(bA)

println_slim(@test bAA1 ≈ bAA2)

## 

sbA = size(bA)
@info("Test batched rrule")
for ntest = 1:30
   local bA, bA2
   local bUU, u
   bA = randn(sbA)
   bU = randn(sbA)
   _BB(t) = bA + t * bU
   bA2 = evaluate(basis2, bA)
   u = randn(size(bA2))
   F(t) = dot(u, evaluate(basis2, _BB(t)))
   dF(t) = begin
      val, pb = rrule(evaluate, basis2, _BB(t))
      ∂BB = pb(u)[3]
      return sum(dot(∂BB[i], bU[i]) for i = 1:length(bU))
   end
   print_tf(@test fdtest(F, dF, 0.0; verbose=false))
end
println() 

##

@info("Test pullback2")

for ntest = 1:30
   local A, AA, Δ, Δ², uA, uΔ, F, dF, basis2 

   basis2 = SparseSymmProd(spec)
   A = randn(2*M+1)
   AA = basis2(A)
   Δ = randn(length(AA)) ./ (1:length(AA))
   Δ² = randn(length(A)) ./ (1:length(A))
   uA = randn(length(A)) ./ (1:length(A))
   uΔ = randn(length(AA)) ./ (1:length(AA))

   F(t) = dot(Δ², ET.pullback(Δ + t * uΔ, basis2, A + t * uA))

   dF(t) = begin 
      val, pb = rrule(ET.pullback, Δ + t * uΔ, basis2,  A + t * uA)
      _, ∇_Δ, _, ∇_A = pb(Δ²)
      return dot(∇_Δ, uΔ) + dot(∇_A, uA)
   end
   print_tf(@test fdtest(F, dF, 0.0; verbose=false))
end
println() 

##

using LinearAlgebra: Diagonal
sbA = size(bA)
@info("Test batched double-pullback")
for ntest = 1:30
   local bA, bA2, bUU, u, Δ, Δ², uΔ, uA 
   bA = randn(sbA)
   bAA = basis2(bA)
   Δ = randn(size(bAA)) /  Diagonal(1:size(bAA, 2))
   Δ² = randn(size(bA)) /  Diagonal(1:size(bA, 2))
   uA = randn(size(bA)) /  Diagonal(1:size(bA, 2))
   uΔ = randn(size(bAA)) / Diagonal(1:size(bAA, 2))

   _Δ(t) = Δ + t * uΔ   
   _X(t) = bA + t * uA

   F(t) = dot(Δ², ET.pullback(_Δ(t), basis2, _X(t)))
   dF(t) = begin
      val, pb = rrule(ET.pullback, _Δ(t), basis2, _X(t))
      _, ∇_Δ, _, ∇_A = pb(Δ²)
      return dot(∇_Δ, uΔ) + dot(∇_A, uA)
   end
   F(0.0)
   dF(0.0)
   print_tf(@test fdtest(F, dF, 0.0; verbose=false))
end
println() 


##


@info("Testing basic pushforward")

for ntest = 1:10 
   local M, nX, spec, A, basis, AA1, AA2 
   M = rand(4:7)
   BO = rand(2:5)
   spec = Test_ACE.generate_SO2_spec(BO, 2*M)
   A = randn(Float64, 2*M+1)
   ΔA = randn(length(A))

   basis = SparseSymmProd(spec)
   AA1 = basis(A)
   AA2, ∂AA2 = ET.pushforward(basis, A, ΔA)
   print_tf( @test AA1 ≈ AA2 )

   u = randn(length(AA1)) ./ (1:length(AA1))
   F(t) = dot(u, basis(A + t * ΔA))
   print_tf(@test fdtest(F, t -> dot(u, ∂AA2), 0.0; verbose=false))
end 
println() 

##

@info("Testing batched basic pushforward")

for ntest = 1:10 
   local M, nX, spec, A, basis, AA1, AA2 
   M = rand(4:7)
   BO = rand(2:5)
   spec = Test_ACE.generate_SO2_spec(BO, 2*M)
   basis = SparseSymmProd(spec)

   nX = rand(6:12)
   A = randn(Float64, nX, 2*M+1)
   ΔA = randn(size(A))

   AA1 = basis(A)
   AA2, ∂AA2 = ET.pushforward(basis, A, ΔA)
   print_tf( @test AA1 ≈ AA2 )

   u = randn(size(AA1))
   F(t) = dot(u, basis(A + t * ΔA))
   print_tf(@test fdtest(F, t -> dot(u, ∂AA2), 0.0; verbose=false))
end 
println() 

##



#= 
# TODO: revive this test 

@info("Testing lux interface")

@info("Test consistency of lux and basis")
l_basis2 = P4ML.lux(basis2)
ps, st = Lux.setup(MersenneTwister(1234), l_basis2)
l_AA2, _ = l_basis2(bA, ps, st)
println_slim(@test l_AA2 ≈ basis2(bA))

println()
=# 

##

@info("Testing KA integration for SparseSymmProd")
include(joinpath(@__DIR__(), "..", "test_utils", "utils_gpu.jl"))

for itest = 1:10 
   local ORD, M, spec, A, AA1, AA2, AA3, AA4 

   ORD = mod1(itest, 3) + 1
   M = rand(20:30)

   spec = Test_ACE.generate_SO2_spec(ORD, 2*M)
   basis = SparseSymmProd(spec)

   A = randn(2*M+1)
   A32 = Float32.(A)
   A_gpu = gpu(A32) 

   # FLoat64 tests, CPU only 
   AA1 = basis(A)
   AA2 = similar(AA1) 
   ET.ka_evaluate!(AA2, basis, A)
   print_tf(@test AA1 ≈ AA2)

   # Float32 tests, CPU and GPU 
   AA3 = Float32.(similar(AA1))
   AA4 = gpu(similar(AA3))
   ET.ka_evaluate!(AA3, basis, A32)
   ET.ka_evaluate!(AA4, basis, A_gpu, gpu.(basis.specs))
   print_tf(@test Float32.(AA1) ≈ AA3)
   print_tf(@test Float32.(AA1) ≈ Array(AA4))
end 
println() 
# TODO: unclear whether many of these tests are really needed 
#       they are already included in the individual layer tests 
#       the frules might be the only ones to keep.
#

# Tests for ForwardDiff compatibility of ACE basis functions
# This verifies that ForwardDiff.jacobian works through the ACE pipeline

using Test, EquivariantTensors, ForwardDiff
using EquivariantTensors: PooledSparseProduct, SparseSymmProd, evaluate, evaluate!,
         _generate_input, _generate_input_1
using ACEbase.Testing: fdtest, println_slim, print_tf
using LinearAlgebra: dot

ET = EquivariantTensors

# Helper to generate pooled sparse product basis
function _generate_basis(; order=3, len = 50)
   NN = [ rand(10:30) for _ = 1:order ]
   spec = sort([ ntuple(t -> rand(1:NN[t]), order) for _ = 1:len])
   return PooledSparseProduct(spec)
end

# Helper to generate symmetric product basis
function _generate_symm_basis(; max_order=3, len_per_order=10)
   n_inputs = 15
   spec = Vector{NTuple{N, Int} where N}()
   for ord = 1:max_order
      for _ = 1:len_per_order
         push!(spec, ntuple(_ -> rand(1:n_inputs), ord))
      end
   end
   return SparseSymmProd(spec)
end

##

@info("Testing ForwardDiff.Dual propagation through PooledSparseProduct")

for ntest = 1:10
   local order, basis, BB, nX
   order = mod1(ntest, 4)
   basis = _generate_basis(; order=order)
   BB = _generate_input(basis)
   nX = size(BB[1], 1)

   # Test that Dual numbers pass through
   BB_dual = ntuple(i -> ForwardDiff.Dual.(BB[i], one(eltype(BB[i]))), order)
   A_dual = evaluate(basis, BB_dual)
   @test eltype(A_dual) <: ForwardDiff.Dual
   @test ForwardDiff.value.(A_dual) ≈ evaluate(basis, BB)
   print_tf(@test true)
end
println()

##

@info("Testing ForwardDiff.jacobian on PooledSparseProduct")

for ntest = 1:10
   local order, basis, BB, nX
   order = mod1(ntest, 3)
   basis = _generate_basis(; order=order, len=30)
   BB = _generate_input(basis)
   nX = size(BB[1], 1)

   # Flatten inputs for jacobian
   BB_flat = vcat([vec(BB[i]) for i in 1:order]...)
   sizes = [size(BB[i]) for i in 1:order]
   lens = [length(BB[i]) for i in 1:order]

   function _unflatten(x)
      idx = 0
      result = ntuple(order) do i
         mat = reshape(x[idx+1:idx+lens[i]], sizes[i])
         idx += lens[i]
         mat
      end
      return result
   end

   f(x) = evaluate(basis, _unflatten(x))

   # Compute Jacobian using ForwardDiff
   J = ForwardDiff.jacobian(f, BB_flat)

   # Verify against finite differences
   ε = 1e-5
   J_fd = zeros(length(basis), length(BB_flat))
   for j in 1:length(BB_flat)
      BB_plus = copy(BB_flat)
      BB_plus[j] += ε
      BB_minus = copy(BB_flat)
      BB_minus[j] -= ε
      J_fd[:, j] = (f(BB_plus) - f(BB_minus)) / (2ε)
   end

   print_tf(@test J ≈ J_fd rtol=1e-5)
end
println()

##

@info("Testing ForwardDiff.Dual propagation through SparseSymmProd")

for ntest = 1:10
   local basis, A
   basis = _generate_symm_basis(max_order=mod1(ntest, 3))
   n_inputs = 15
   A = randn(n_inputs)

   # Test that Dual numbers pass through
   A_dual = ForwardDiff.Dual.(A, one(eltype(A)))
   AA_dual = evaluate(basis, A_dual)
   @test eltype(AA_dual) <: ForwardDiff.Dual
   @test ForwardDiff.value.(AA_dual) ≈ evaluate(basis, A)
   print_tf(@test true)
end
println()

##

@info("Testing ForwardDiff.jacobian on SparseSymmProd")

for ntest = 1:10
   local basis, A
   basis = _generate_symm_basis(max_order=mod1(ntest, 3))
   n_inputs = 15
   A = randn(n_inputs)

   f(x) = evaluate(basis, x)

   # Compute Jacobian using ForwardDiff
   J = ForwardDiff.jacobian(f, A)

   # Verify against finite differences
   ε = 1e-5
   J_fd = zeros(length(basis), length(A))
   for j in 1:length(A)
      A_plus = copy(A)
      A_plus[j] += ε
      A_minus = copy(A)
      A_minus[j] -= ε
      J_fd[:, j] = (f(A_plus) - f(A_minus)) / (2ε)
   end

   print_tf(@test J ≈ J_fd rtol=1e-5)
end
println()

##

@info("Testing frule for PooledSparseProduct")

using ChainRulesCore

for ntest = 1:10
   local order, basis, BB, ΔBB
   order = mod1(ntest, 4)
   basis = _generate_basis(; order=order)
   BB = _generate_input(basis)
   ΔBB = ntuple(i -> randn(size(BB[i])), order)

   # Use frule
   y, Δy = ChainRulesCore.frule((nothing, nothing, ΔBB), evaluate, basis, BB)

   # Compare with finite difference
   ε = 1e-7
   BB_plus = ntuple(i -> BB[i] + ε * ΔBB[i], order)
   BB_minus = ntuple(i -> BB[i] - ε * ΔBB[i], order)
   Δy_fd = (evaluate(basis, BB_plus) - evaluate(basis, BB_minus)) / (2ε)

   @test y ≈ evaluate(basis, BB)
   print_tf(@test Δy ≈ Δy_fd rtol=1e-5)
end
println()

##

@info("Testing frule for SparseSymmProd")

for ntest = 1:10
   local basis, A, ΔA
   basis = _generate_symm_basis(max_order=mod1(ntest, 3))
   n_inputs = 15
   A = randn(n_inputs)
   ΔA = randn(n_inputs)

   # Use frule
   y, Δy = ChainRulesCore.frule((nothing, nothing, ΔA), evaluate, basis, A)

   # Compare with finite difference
   ε = 1e-7
   Δy_fd = (evaluate(basis, A + ε * ΔA) - evaluate(basis, A - ε * ΔA)) / (2ε)

   @test y ≈ evaluate(basis, A)
   print_tf(@test Δy ≈ Δy_fd rtol=1e-5)
end
println()

##

@info("All ForwardDiff tests completed!")

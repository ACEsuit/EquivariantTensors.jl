
using SparseArrays: SparseMatrixCSC 
import LinearAlgebra: mul! 
import MLDataDevices: AbstractGPUDevice
using GPUArraysCore: AbstractGPUArray 
import Adapt  

struct DevSparseMatrixCSR{VECI, VECV}
   m::Int              # Number of rows
   n::Int              # Number of columns
   rowptr::VECI        # Row i is in rowptr[i]:(rowptr[i+1]-1)
   colval::VECI        # Col indices of stored values
   nzval::VECV         # Stored values, typically nonzeros
end


function DevSparseMatrixCSR(A::SparseMatrixCSC, dev = identity)
   At = SparseMatrixCSC(permutedims(A)) 
   return DevSparseMatrixCSR(A.m, A.n, dev(At.colptr), dev(At.rowval), dev(At.nzval))
end

_floatT(T::Type{<: AbstractFloat}, A::DevSparseMatrixCSR) = 
   DevSparseMatrixCSR(A.m, A.n, A.rowptr, A.colval, _floatT(T, A.nzval))

Base.size(A::DevSparseMatrixCSR) = (A.m, A.n)
Base.size(A::DevSparseMatrixCSR, i::Integer) = size(A)[i]
Base.eltype(A::DevSparseMatrixCSR) = eltype(A.nzval)

# this is not really used (also no unit tests), 
# but it can be useful for debugging
function Base.getindex(A::DevSparseMatrixCSR, i::Int, j::Int)
   @assert 1 <= i <= A.m
   @assert 1 <= j <= A.n
   for idx = A.rowptr[i]:(A.rowptr[i+1]-1)
      if A.colval[idx] == j
         return A.nzval[idx]
      end
   end
   return zero(eltype(A.nzval))
end

function mul(A::DevSparseMatrixCSR, b::AbstractVector)
   m, n = A.m, A.n 
   x = similar(b, (m,))
   fill!(x, zero(eltype(x)))
   mul!(reshape(x, (:, 1)), A, reshape(b, (:, 1))) 
   return x
end

function mul(A::DevSparseMatrixCSR, B::AbstractMatrix)
   TX = typeof(zero(eltype(A.nzval)) * zero(eltype(B)))
   X = similar(B, TX, (size(A, 1), size(B, 2)))
   return mul!(X, A, B)
end

function mul(A::AbstractMatrix, B::DevSparseMatrixCSR, op = *)
   TX = typeof( op(zero(eltype(A)), zero(eltype(B.nzval))) ) 
   X = similar(A, TX, (size(A, 1), size(B, 2)))
   return mul!(X, A, B, op)
end



function mul!(X::AbstractMatrix, A::DevSparseMatrixCSR, B::AbstractMatrix)

   @kernel function _mul_ka_sparse!(X, B, rowptr, colval, nzval)
      row, jB = @index(Global, NTuple)
      
      for idx = rowptr[row]:(rowptr[row+1]-1)
         col = colval[idx]
         X[row, jB] += nzval[idx] * B[col, jB]
      end

      nothing 
   end
   
   m, n = A.m, A.n 
   rowptr = A.rowptr
   colval = A.colval
   nzval = A.nzval

   @assert size(B, 1) == n
   @assert size(X, 1) == m
   @assert size(X, 2) == size(B, 2)
   fill!(X, zero(eltype(X)))

   kernel! = _mul_ka_sparse!(KernelAbstractions.get_backend(X))
   kernel!(X, B, rowptr, colval, nzval; ndrange = (m, size(B, 2)))
   return X
end 



function mul!(X::AbstractMatrix, A::AbstractMatrix, B::DevSparseMatrixCSR, op=*)
   # B = [ row1 ; row2 ; ... ] 
   
   @kernel function _mul_ka_dense_sparse!(X, A, rowptr, colval, nzval)
      # X = A * B 
      rowA, rowB = @index(Global, NTuple)
      
      for idx = rowptr[rowB]:(rowptr[rowB+1]-1)
         colB = colval[idx]
         # This needs to be atomic because X[rowA, colB] is updated 
         # in parallel; to avoid this, we need to switch to a CSC format 
         # we can achieve this by storing A2Bmaps in both formats, one for the 
         # forward pass, the other for the backward pass.
         @atomic X[rowA, colB] += op(A[rowA, rowB], nzval[idx])
      end

      nothing 
   end

   m, n = size(A)
   k = size(B, 2)
   rowptr = B.rowptr
   colval = B.colval
   nzval = B.nzval

   @assert size(B) == (n, k)
   @assert size(X) == (m, k)
   fill!(X, zero(eltype(X)))

   kernel! = _mul_ka_dense_sparse!(KernelAbstractions.get_backend(X))
   kernel!(X, A, rowptr, colval, nzval; ndrange = (m, size(B, 1)))
   return X
end 


# NOTE: If AA comes in the format of nnodes x nfeatures then we actually need 
#       a multiplication of the form AA * 𝒞' - this is likely more performant 
#       if done by keeping AA intact and then storing C' in CSC.
#       (something to consider in the future when debugging performance)


using SparseArrays: SparseMatrixCSC 
import LinearAlgebra: mul! 
using MLDataDevices: AbstractGPUDevice
using GPUArraysCore: AbstractGPUArray 

struct DevSparseMatrixCSR{VECI, VECV}
   m::Int              # Number of rows
   n::Int              # Number of columns
   rowptr::VECI        # Row i is in rowptr[i]:(rowptr[i+1]-1)
   colval::VECI        # Col indices of stored values
   nzval::VECV         # Stored values, typically nonzeros
end


function DevSparseMatrixCSR(A::SparseMatrixCSC, dev = identity)
   At = SparseMatrixCSC(transpose(A)) 
   return DevSparseMatrixCSR(A.m, A.n, dev(At.colptr), dev(At.rowval), dev(At.nzval))
end

(dev::AbstractGPUDevice)(A::SparseMatrixCSC) = DevSparseMatrixCSR(A, dev)
(T::Type{AbstractGPUArray})(A::SparseMatrixCSC) = DevSparseMatrixCSR(A, T)
Base.convert(T::Type{<: AbstractGPUArray}, A::SparseMatrixCSC) = DevSparseMatrixCSR(A, T)

Base.size(A::DevSparseMatrixCSR) = (A.m, A.n)

function mul(A::DevSparseMatrixCSR, b::AbstractVector)
   m, n = A.m, A.n 
   x = similar(b, (m,))
   fill!(x, zero(eltype(x)))
   mul!(reshape(x, (:, 1)), A, reshape(b, (:, 1))) 
   return x
end

function mul(A::DevSparseMatrixCSR, B::AbstractMatrix)
   m, n = A.m, A.n 
   X = similar(B, (m, size(B, 2)))
   return mul!(X, A, B)
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

# NOTE: If AA comes in the format of nnodes x nfeatures then we actually need 
#       a multiplication of the form AA * 𝒞' - this is likely more performant 
#       if done by keeping AA intact and then storing C' in CSC.
#       (something to consider in the future when debugging performance)

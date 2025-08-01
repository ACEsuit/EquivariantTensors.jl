
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

# note the __arr2dev__ is used to workaround the fact that arrays of non-number 
# bitstypes are not automagically converted to GPUArrays. This seems a bug 
# or missing feature either in MLDataDevices or in the GPU Arrays packages. 

# Adapt.adapt(::AbstractGPUDevice, A::DevSparseMatrixCSR) = 
#       DevSparseMatrixCSR(A.m, A.n, dev(A.rowptr), dev(A.colval), 
#                          __arr2dev__(dev, A.nzval))

# __arr2dev__(dev::AbstractGPUDevice, A::Vector{<: Number}) = dev(A) 

# function __arr2dev__(dev::AbstractGPUDevice, A::Vector{SVector{N, T}}) where {N, T}
#    Amat = collect(reinterpret(reshape, T, A))
#    Amatdev = dev(Amat) 
#    return reinterpret(SVector{N, eltype(Amatdev)}, Amatdev)
# end

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!   this is type piracy; we should not do this             !!!
# !!!   this needs to be thought about very very carefully     !!!
# (dev::AbstractGPUDevice)(A::SparseMatrixCSC) = DevSparseMatrixCSR(A, dev)
# (T::Type{AbstractGPUArray})(A::SparseMatrixCSC) = DevSparseMatrixCSR(A, T)
# Base.convert(T::Type{<: AbstractGPUArray}, A::SparseMatrixCSC) = DevSparseMatrixCSR(A, T)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


Base.size(A::DevSparseMatrixCSR) = (A.m, A.n)
Base.size(A::DevSparseMatrixCSR, i::Integer) = size(A)[i]
Base.eltype(A::DevSparseMatrixCSR) = eltype(A.nzval)

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

function mul(A::AbstractMatrix, B::DevSparseMatrixCSR)
   TX = typeof(zero(eltype(A.nzval)) * zero(eltype(B)))
   X = similar(A, TX, (size(A, 1), size(B, 2)))
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


function mul!(X::AbstractMatrix, A::AbstractMatrix, B::DevSparseMatrixCSR)
   # B = [ row1 ; row2 ; ... ] 

   @kernel function _mul_ka_dense_sparse!(X, A, rowptr, colval, nzval)
      # X = A * B 
      rowA, rowB = @index(Global, NTuple)
      
      for idx = rowptr[rowB]:(rowptr[rowB+1]-1)
         colB = colval[idx]
         X[rowA, colB] += A[rowA, rowB] * nzval[idx]
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
   kernel!(X, A, rowptr, colval, nzval; ndrange = (m, size(B, 2)))
   return X
end 


# NOTE: If AA comes in the format of nnodes x nfeatures then we actually need 
#       a multiplication of the form AA * 𝒞' - this is likely more performant 
#       if done by keeping AA intact and then storing C' in CSC.
#       (something to consider in the future when debugging performance)

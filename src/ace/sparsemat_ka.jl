
using SparseArrays: SparseMatrixCSC 
import LinearAlgebra: mul! 

import Adapt
using Adapt: adapt 


"""
   struct SparseMatCSX

Sparse matrix format that is stored in both CSR and CSC, and can be transferred 
to GPU devices. Matrix multiplication via KernelAbstractions.jl for GPU support
and uses whichever format (CSR, CSC) is most suitable for a given operation. 
"""
struct SparseMatCSX{VECI, VECV}
   m::Int              # Number of rows
   n::Int              # Number of columns
   # ---------- CSR format 
   rowptr::VECI        # Row i is in rowptr[i]:(rowptr[i+1]-1)
   colval::VECI        # Col indices of stored values
   nzval_csr::VECV     # Stored values, typically nonzeros
   # ---------- CSC format
   colptr::VECI        # Col j is in colptr[j]:(colptr[j+1]-1)
   rowval::VECI        # Row indices of stored values
   nzval_csc::VECV     # Stored values, typically nonzeros
end


function SparseMatCSX(A::SparseMatrixCSC, dev = identity)
   At = SparseMatrixCSC(permutedims(A)) 
   return SparseMatCSX(A.m, A.n, 
               dev(At.colptr), dev(At.rowval), dev(At.nzval), 
               dev(A.colptr),  dev(A.rowval),  dev(A.nzval)  )
end

function Adapt.adapt_structure(to, X::SparseMatCSX) 
   SparseMatCSX( X.m, X.n,  
                 adapt(to, X.rowptr), adapt(to, X.colval), adapt(to, X.nzval_csr), 
                 adapt(to, X.colptr), adapt(to, X.rowval), adapt(to, X.nzval_csc) )
end


_floatT(T::Type{<: AbstractFloat}, A::SparseMatCSX) = 
   SparseMatCSX(A.m, A.n, A.rowptr, A.colval, _floatT(T, A.nzval_csr), 
                             A.colptr, A.rowval, A.nzval_csc)

Base.size(A::SparseMatCSX) = (A.m, A.n)
Base.size(A::SparseMatCSX, i::Integer) = size(A)[i]
Base.eltype(A::SparseMatCSX) = eltype(A.nzval_csr)

# this is not really used (also no unit tests), 
# but it can be useful for debugging
function Base.getindex(A::SparseMatCSX, i::Int, j::Int)
   @assert 1 <= i <= A.m
   @assert 1 <= j <= A.n
   for idx = A.rowptr[i]:(A.rowptr[i+1]-1)
      if A.colval[idx] == j
         return A.nzval_csr[idx]
      end
   end
   return zero(eltype(A.nzval_csr))
end

function mul(A::SparseMatCSX, b::AbstractVector)
   m, n = A.m, A.n 
   TX = typeof(zero(eltype(A)) * zero(eltype(b)))
   x = similar(b, TX, (m,))
   fill!(x, zero(TX))
   mul!(reshape(x, (:, 1)), A, reshape(b, (:, 1))) 
   return x
end

function mul(A::SparseMatCSX, B::AbstractMatrix)
   TX = typeof(zero(eltype(A)) * zero(eltype(B)))
   X = similar(B, TX, (size(A, 1), size(B, 2)))
   return mul!(X, A, B)
end

function mul(A::AbstractMatrix, B::SparseMatCSX, op = *)
   TX = typeof( op(zero(eltype(A)), zero(eltype(B))) )
   X = similar(A, TX, (size(A, 1), size(B, 2)))
   return mul!(X, A, B, op)
end

#
# X = A * B where A is sparse, B is dense 
# this is easiest to implement in CSR format 
#
function mul!(X::AbstractMatrix, A::SparseMatCSX, B::AbstractMatrix, op=*)

   @kernel function _mul_ka_sparse_dense!(X, B, rowptr, colval, nzval_csr)
      row, jB = @index(Global, NTuple)
      
      for idx = rowptr[row]:(rowptr[row+1]-1)
         col = colval[idx]
         X[row, jB] += nzval_csr[idx] * B[col, jB]
      end

      nothing 
   end
   
   m, n = A.m, A.n 
   @assert size(B, 1) == n
   @assert size(X, 1) == m
   @assert size(X, 2) == size(B, 2)
   fill!(X, zero(eltype(X)))

   kernel! = _mul_ka_sparse_dense!(KernelAbstractions.get_backend(X))
   kernel!(X, B, A.rowptr, A.colval, A.nzval_csr; ndrange = (m, size(B, 2)))
   return X
end 


#
# X = A * B where A is dense, B is sparse 
# this is easiest to implement in CSC format 
#

function mul!(X::AbstractMatrix, A::AbstractMatrix, B::SparseMatCSX, op=*)
   # B = [ col1 | col2 | ... ] 

   @kernel function _mul_ka_dense_sparse!(X, A, colptr, rowval, nzval_csc)
      # X = A * B 
      rowA, colB = @index(Global, NTuple)

      for idx = colptr[colB]:(colptr[colB+1]-1)
         rowB = rowval[idx]
         X[rowA, colB] += op(A[rowA, rowB], nzval_csc[idx])
      end

      nothing 
   end
   

   m, n = size(A)
   k = size(B, 2)

   @assert size(B) == (n, k)
   @assert size(X) == (m, k)
   fill!(X, zero(eltype(X)))

   kernel! = _mul_ka_dense_sparse!(KernelAbstractions.get_backend(X))
   kernel!(X, A, B.colptr, B.rowval, B.nzval_csc; ndrange = (m, k))
   return X
end 



# NOTE: If AA comes in the format of nnodes x nfeatures then we actually need 
#       a multiplication of the form AA * 𝒞' - this is likely more performant 
#       if done by keeping AA intact and then storing C' in CSC.
#       (something to consider in the future when debugging performance)

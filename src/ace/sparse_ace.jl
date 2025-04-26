

using SparseArrays: SparseMatrixCSC     
import Polynomials4ML


struct SparseACE{T, TA, TAA}
   abasis::TA
   aabasis::TAA
   A2Bmap::SparseMatrixCSC{T, Int}
   # ------- 
   meta::Dict{String, Any}
end

Base.length(tensor::SparseEquivTensor) = size(tensor.A2Bmap, 1) 


function evaluate!(B, _AA, tensor::SparseEquivTensor{T}, Rnl, Ylm) where {T}
   # evaluate the A basis
   TA = promote_type(T, eltype(Rnl), eltype(eltype(Ylm)))
   A = zeros(TA, length(tensor.abasis))
   P4ML.evaluate!(A, tensor.abasis, (Rnl, Ylm))

   # evaluate the AA basis
   # _AA = zeros(TA, length(tensor.aabasis))     # use Bumper here
   P4ML.evaluate!(_AA, tensor.aabasis, A)
   # project to the actual AA basis 
   proj = tensor.aabasis.projection
   AA = _AA[proj]     # use Bumper here, or view; needs experimentation. 

   # evaluate the coupling coefficients
   # B = tensor.A2Bmap * AA
   mul!(B, tensor.A2Bmap, AA)   

   return B, (_AA = _AA, )
end

function whatalloc(::typeof(evaluate!), tensor::SparseEquivTensor, Rnl, Ylm)
   TA = promote_type(eltype(Rnl), eltype(eltype(Ylm)))
   TB = promote_type(TA, eltype(tensor.A2Bmap))
   return (TB, length(tensor),), (TA, length(tensor.aabasis),)
end

function evaluate(tensor::SparseEquivTensor, Rnl, Ylm)
   allocinfo = whatalloc(evaluate!, tensor, Rnl, Ylm)
   B = zeros(allocinfo[1]...)
   AA = zeros(allocinfo[2]...)
   return evaluate!(B, AA, tensor, Rnl, Ylm)
end


# ---------


function pullback!(∂Rnl, ∂Ylm, 
                   ∂B, tensor::SparseEquivTensor, Rnl, Ylm, 
                   intermediates)
   _AA = intermediates._AA
   proj = tensor.aabasis.projection
   T_∂AA = promote_type(eltype(∂B), eltype(tensor.A2Bmap))
   T_∂A = promote_type(T_∂AA, eltype(_AA))

   @no_escape begin 
   #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                           
   # ∂Ei / ∂AA = ∂Ei / ∂B * ∂B / ∂AA = (WB[i_z0]) * A2Bmap
   # ∂AA = tensor.A2Bmap' * ∂B   
   ∂AA = @alloc(T_∂AA, size(tensor.A2Bmap, 2))
   mul!(∂AA, tensor.A2Bmap', ∂B)
   _∂AA = @alloc(T_∂AA, length(_AA))
   fill!(_∂AA, zero(T_∂AA))
   _∂AA[proj] = ∂AA

   # ∂Ei / ∂A = ∂Ei / ∂AA * ∂AA / ∂A = pullback(aabasis, ∂AA)
   ∂A = @alloc(T_∂A, length(tensor.abasis))
   P4ML.unsafe_pullback!(∂A, _∂AA, tensor.aabasis, _AA)
   
   # ∂Ei / ∂Rnl, ∂Ei / ∂Ylm = pullback(abasis, ∂A)
   P4ML.pullback!((∂Rnl, ∂Ylm), ∂A, tensor.abasis, (Rnl, Ylm))

   #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   end # no_escape

   return ∂Rnl, ∂Ylm
end

function whatalloc(::typeof(pullback!),  
                   ∂B, tensor::SparseEquivTensor{T}, Rnl, Ylm, 
                   intermediates) where {T} 
   TA = promote_type(T, eltype(intermediates._AA), eltype(∂B), 
                     eltype(Rnl), eltype(eltype(Ylm)))
   return (TA, size(Rnl)...), (TA, size(Ylm)...)   
end

function pullback(∂B, tensor::SparseEquivTensor{T}, Rnl, Ylm, 
                           intermediates) where {T} 
   alc_∂Rnl, alc_∂Ylm = whatalloc(pullback!, ∂B, tensor, Rnl, Ylm, intermediates)
   ∂Rnl = zeros(alc_∂Rnl...)
   ∂Ylm = zeros(alc_∂Ylm...)
   return pullback!(∂Rnl, ∂Ylm, ∂B, tensor, Rnl, Ylm, intermediates)
end

# ----------------------------------------
#  utilities 

"""
Get the specification of the BBbasis as a list (`Vector`) of vectors of `@NamedTuple{n::Int, l::Int}`.

### Parameters 

* `tensor` : a SparseEquivTensor, possibly from ACEModel
"""
function get_nnll_spec(tensor::SparseEquivTensor{T}) where {T}
   _nl(bb) = [(n = b.n, l = b.l) for b in bb]
   # assume the new ACE model NEVER has the z channel
   spec = tensor.aabasis.meta["AA_spec"]
   nBB = size(tensor.A2Bmap, 1)
   nnll_list = Vector{NT_NL_SPEC}[]
   for i in 1:nBB
      AAidx_nnz = tensor.A2Bmap[i, :].nzind
      bbs = spec[AAidx_nnz]
      @assert all([bb == _nl(bbs[1]) for bb in _nl.(bbs)])
      push!(nnll_list, _nl(bbs[1]))
   end
   @assert length(nnll_list) == nBB
   return nnll_list
end



# ----------------------------------------
#  experimental pushforwards 

function _pfwd(tensor::SparseEquivTensor{T}, Rnl, Ylm, ∂Rnl, ∂Ylm) where {T}
   A, ∂A = _pfwd(tensor.abasis, (Rnl, Ylm), (∂Rnl, ∂Ylm))
   _AA, _∂AA = _pfwd(tensor.aabasis, A, ∂A)

   # project to the actual AA basis 
   proj = tensor.aabasis.projection
   AA = _AA[proj]  
   ∂AA = _∂AA[proj, :]

   # evaluate the coupling coefficients
   B = tensor.A2Bmap * AA 
   ∂B = tensor.A2Bmap * ∂AA 
   return B, ∂B 
end


function _pfwd(abasis::Polynomials4ML.PooledSparseProduct{2}, RY, ∂RY) 
   R, Y = RY 
   TA = typeof(R[1] * Y[1])
   ∂R, ∂Y = ∂RY
   ∂TA = typeof(R[1] * ∂Y[1] + ∂R[1] * Y[1])

   # check lengths 
   nX = size(R, 1)
   @assert nX == size(R, 1) == size(∂R, 1) == size(Y, 1) == size(∂Y, 1)

   A = zeros(TA, length(abasis.spec))
   ∂A = zeros(∂TA, size(∂R, 1), length(abasis.spec))

   for i = 1:length(abasis.spec)
      @inbounds begin 
         n1, n2 = abasis.spec[i]
         ai = zero(TA)
         @simd ivdep for α = 1:nX 
            ai += R[α, n1] * Y[α, n2]
            ∂A[α, i] = R[α, n1] * ∂Y[α, n2] + ∂R[α, n1] * Y[α, n2]
         end 
         A[i] = ai
      end 
   end 
   return A, ∂A
end 


function _pfwd(aabasis::Polynomials4ML.SparseSymmProdDAG, A, ∂A)
   n∂ = size(∂A, 1)
   num1 = aabasis.num1 
   nodes = aabasis.nodes 
   AA = zeros(eltype(A), length(nodes))
   T∂AA = typeof(A[1] * ∂A[1])
   ∂AA = zeros(T∂AA, length(nodes), size(∂A, 1))
   for i = 1:num1 
      AA[i] = A[i] 
      for α = 1:n∂
         ∂AA[i, α] = ∂A[α, i]
      end
   end 
   for iAA = num1+1:length(nodes)
      n1, n2 = nodes[iAA]
      AA_n1 = AA[n1]
      AA_n2 = AA[n2]
      AA[iAA] = AA_n1 * AA_n2
      for α = 1:n∂
         ∂AA[iAA, α] = AA_n2 * ∂AA[n1, α] + AA_n1 * ∂AA[n2, α]
      end
   end
   return AA, ∂AA
end




# can we ignore the level function here? 
function _make_A_spec(AA_spec, level)
   NT_NLM = NamedTuple{(:n, :l, :m), Tuple{Int, Int, Int}}
   A_spec = NT_NLM[]
   for bb in AA_spec 
      append!(A_spec, bb)
   end
   A_spec = unique(A_spec)
   A_spec_level = [ level(b) for b in A_spec ]
   p = sortperm(A_spec_level)
   A_spec = A_spec[p]
   return A_spec
end 

# TODO: this should go into sphericart or P4ML 
function _make_Y_spec(maxl::Integer)
   NT_LM = NamedTuple{(:l, :m), Tuple{Int, Int}}
   y_spec = NT_LM[] 
   for i = 1:P4ML.SpheriCart.sizeY(maxl)
      l, m = P4ML.SpheriCart.idx2lm(i)
      push!(y_spec, (l = l, m = m))
   end
   return y_spec 
end

function _make_idx_A_spec(A_spec, r_spec, y_spec)
   inv_r_spec = _inv_list(r_spec)
   inv_y_spec = _inv_list(y_spec)
   A_spec_idx = [ (inv_r_spec[(n=b.n, l=b.l)], inv_y_spec[(l=b.l, m=b.m)]) 
                  for b in A_spec ]
   return A_spec_idx                  
end

function _make_idx_AA_spec(AA_spec, A_spec) 
   inv_A_spec = _inv_list(A_spec)
   AA_spec_idx = [ [ inv_A_spec[b] for b in bb ] for bb in AA_spec ]
   sort!.(AA_spec_idx)
   return AA_spec_idx
end 


function _generate_ace_model(rbasis, Ytype::Symbol, AA_spec::AbstractVector, 
                             Vref, 
                             level = TotalDegree(), 
                             pair_basis = nothing, 
                             ) 

   # # storing E0s with unit
   # model_meta = Dict{String, Any}("E0s" => deepcopy(E0s))
   model_meta = Dict{String, Any}()

   # generate the coupling coefficients 
   cgen = EquivariantModels.Rot3DCoeffs_real(0)
   AA2BB_map = EquivariantModels._rpi_A2B_matrix(cgen, AA_spec)

   # find which AA basis functions are actually used and discard the rest 
   keep_AA_idx = findall(sum(abs, AA2BB_map; dims = 1)[:] .> 0)
   AA_spec = AA_spec[keep_AA_idx]
   AA2BB_map = AA2BB_map[:, keep_AA_idx]

   # generate the corresponding A basis spec
   A_spec = _make_A_spec(AA_spec, level)

   # from the A basis we can generate the Y basis since we now know the 
   # maximum l value (though we probably already knew that from r_spec)
   maxl = maximum([ b.l for b in A_spec ])   
   ybasis = _make_Y_basis(Ytype, maxl)
   
   # now we need to take the human-readable specs and convert them into 
   # the layer-readable specs 
   r_spec = rbasis.spec
   y_spec = _make_Y_spec(maxl)

   # get the idx version of A_spec 
   A_spec_idx = _make_idx_A_spec(A_spec, r_spec, y_spec)

   # from this we can now generate the A basis layer                   
   a_basis = Polynomials4ML.PooledSparseProduct(A_spec_idx)
   a_basis.meta["A_spec"] = A_spec  #(also store the human-readable spec)

   # get the idx version of AA_spec
   AA_spec_idx = _make_idx_AA_spec(AA_spec, A_spec) 

   # from this we can now generate the AA basis layer
   aa_basis = Polynomials4ML.SparseSymmProdDAG(AA_spec_idx)
   aa_basis.meta["AA_spec"] = AA_spec  # (also store the human-readable spec)

   tensor = SparseEquivTensor(a_basis, aa_basis, AA2BB_map, 
                              Dict{String, Any}())

   return ACEModel(rbasis._i2z, rbasis, ybasis, 
                   tensor, pair_basis, Vref, 
                   model_meta )
end

# TODO: it is not entirely clear that the `level` is really needed here 
#       since it is implicitly already encoded in AA_spec. We need a 
#       function `auto_level` that generates level automagically from AA_spec.
function ace_model(rbasis, Ytype, AA_spec::AbstractVector, level, 
                   pair_basis, Vref)
   return _generate_ace_model(rbasis, Ytype, AA_spec, Vref, level, pair_basis)
end 

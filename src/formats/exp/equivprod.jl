#
# implementation of an equivariant tensor product
# - should behave exactly like an equivariant linear ACE basis: 
#   input (Rnl, Ylm) => output (B0, B1, ...) 
# -

using StaticArrays: SVector

"""
   EquivariantTensorProduct

Behaves exactly like an equivariant linear ACE basis, but without the pooling. 
Takes as input a tuple of (R, Y) edge embeddings and produces an output that 
is equivalent to an equivariant ACE basis.
"""
struct EquivariantTensorProduct{NL} # make it a Lux container layer
   ranges::NTuple{NL, Vector{Int}}  # ranges[i] = radial indices for the LL[i] output
   LL::NTuple{NL, Int}
   # ----
   meta::Dict{String, Any}
end

function EquivariantTensorProduct(LL, Rnl_spec, Ylm_spec)
   LL = tuple(LL...)

   # For each requested output L, collect the radial indices that contribute
   # to it. The L-output couples a single radial Rnl[iR] to the full angular
   # block Ylm_{L, m}, m = -L..L. If the radial spec carries an `l`, only
   # radials with l == L couple to that block; otherwise (radial keyed on `n`
   # only) every radial contributes to every L. The angular block itself is
   # not indexed here: its location is fixed by L and the (l,m) ordering.
   radial_has_l = (:l in fieldnames(eltype(Rnl_spec)))
   ranges = ntuple(length(LL)) do i
      radial_has_l ? findall(r -> r.l == LL[i], Rnl_spec) :
                     collect(1:length(Rnl_spec))
   end

   # TODO Claude: confirm that the Ylm spec is consistent with the sphericart convention 
   # since we will use this explicitly during evaluation

   meta = Dict{String, Any}("Rnl_spec" => Rnl_spec,
                            "Ylm_spec" => Ylm_spec,
                            "LL" => LL)

   return EquivariantTensorProduct(ranges, LL, meta)
end

# ------ Lux ps and st 

initialparameters(rng::AbstractRNG, bas::EquivariantTensorProduct) = 
         NamedTuple() 

initialstates(rng::AbstractRNG, bas::EquivariantTensorProduct) =
         (  ranges = bas.ranges,
            LL = bas.LL,
         )


# -------- evaluation kernels 

# format of Rnl, Ylm 3-tensor is determined by EdgeEmbedding
# that is the three dimensions are (j_neig, i_node, k_feat)

function evaluate(op::EquivariantTensorProduct,
         Rnl::AbstractArray{T, 3}, Ylm::AbstractArray{T, 3}, ps, st) where {T}
   return ka_evaluate(op, Rnl, Ylm, ps, st)
end


function ka_evaluate(op::EquivariantTensorProduct{NL},
         Rnl::AbstractArray{T, 3}, Ylm::AbstractArray{T, 3}, ps, st
         ) where {NL, T}
   Lmax = maximum(op.LL)
   @assert size(Ylm, 3) >= (Lmax + 1)^2

   # one output array per L in LL. Wrapping L in a Val makes the SVector
   # length (2L+1) a compile-time constant inside _ka_evaluate_L, so each
   # kernel launch is type stable. NB: the element type of the returned tuple
   # 𝔹 still depends on the runtime values op.LL[i], so 𝔹 itself is not
   # concretely inferred (this would require LL to be a type parameter).
   𝔹 = ntuple(i -> _ka_evaluate_L(Rnl, Ylm, op.ranges[i], Val(op.LL[i])), NL)

   KernelAbstractions.synchronize(KernelAbstractions.get_backend(Rnl))
   return 𝔹
end


function _ka_evaluate_L(Rnl::AbstractArray{T, 3}, Ylm::AbstractArray{T, 3},
                        range, ::Val{L}) where {T, L}
   nneig, nnode = size(Rnl, 1), size(Rnl, 2)
   𝔹L = similar(Rnl, SVector{2*L+1, T}, (nneig, nnode, length(range)))
   backend = KernelAbstractions.get_backend(Rnl)
   kernel! = _ka_evaluate_L!(backend)
   kernel!(EquivariantTensorProduct, 𝔹L, Rnl, Ylm, range, L;
           ndrange = (nneig, nnode, length(range)))
   return 𝔹L
end


@kernel function _ka_evaluate_L!(::Type{EquivariantTensorProduct},
         𝔹L,          # abstractarray SVector{2*L+1, T}, to write into
         Rnl, Ylm,    # const abstractarray T, format (j_neig, i_node, k_feat)
         range,       # const abstractvector Int, radial indices for this L
         L::Int)
   j, i, k = @index(Global, NTuple)
   # k selects the feature, range[k] is the radial index iR into Rnl; the
   # angular part is the full L-block Ylm_{L, m}, m = -L..L, gathered below.
   iR = range[k]
   r = Rnl[j, i, iR]
   yL = _extract_yL(eltype(𝔹L), Ylm, j, i, L)
   𝔹L[j, i, k] = r * yL
end


# The sphericart (l, m) -> index mapping is  l^2 + l + m + 1, so the full
# L-block sits at indices L^2+1 .. (L+1)^2, i.e. m = -L..L in order. This
# generated function unrolls the gather into a length-(2L+1) SVector.
@generated function _extract_yL(::Type{SVector{P, T}}, Ylm, j, i, L
                                ) where {P, T}
   vals = [ :(Ylm[j, i, L*L + $t]) for t = 1:P ]
   return :( SVector{P, T}( $(vals...) ) )
end

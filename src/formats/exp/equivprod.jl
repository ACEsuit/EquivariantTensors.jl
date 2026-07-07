#
# implementation of an equivariant tensor product
# - should behave exactly like an equivariant linear ACE basis: 
#   input (Rnl, Ylm) => output (B0, B1, ...) 
# -

using StaticArrays: SVector
using SpheriCart: lm2idx

# NB on type inference: the angular momenta LL are stored as a *field* (their
# values are only known at runtime), not as a type parameter. Evaluation
# returns a tuple 𝔹 whose i-th entry has element type SVector{2*LL[i]+1, T}.
# Because the LL[i] are runtime values, those element types cannot be inferred,
# so 𝔹 is a non-concrete tuple at the call site — even though each per-L kernel
# launch is type stable behind a Val(L) barrier. Promoting LL into the type
# parameter would recover full inference, at the cost of more type churn and
# recompilation across different LL.

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

   # Evaluation indexes the Ylm array directly through SpheriCart's `lm2idx`
   # mapping, so the Ylm_spec must follow that same ordering and cover every
   # (l, m) up to Lmax = maximum(LL). Verify this now.
   _assert_sphericart_spec(Ylm_spec, maximum(LL))

   meta = Dict{String, Any}("Rnl_spec" => Rnl_spec,
                            "Ylm_spec" => Ylm_spec,
                            "LL" => LL)

   return EquivariantTensorProduct(ranges, LL, meta)
end

# Check that `Ylm_spec` lists the (l, m) pairs in the SpheriCart order, i.e.
# the entry at position i satisfies i == lm2idx(l, m), and that it covers all
# l up to `Lmax`. Only the first (Lmax+1)^2 entries are used during evaluation,
# so only those are checked; any extra trailing entries are ignored.
function _assert_sphericart_spec(Ylm_spec, Lmax)
   nY = (Lmax + 1)^2
   length(Ylm_spec) >= nY ||
      error("EquivariantTensorProduct: Ylm_spec has $(length(Ylm_spec)) \
             entries but needs at least $nY to cover L up to $Lmax.")
   for i = 1:nY
      y = Ylm_spec[i]
      idx = lm2idx(y.l, y.m)
      idx == i ||
         error("EquivariantTensorProduct: Ylm_spec is inconsistent with the \
                sphericart convention: entry $i is (l=$(y.l), m=$(y.m)) but \
                its sphericart index is $idx.")
   end
   return nothing
end

# ------ Lux ps and st 

initialparameters(rng::AbstractRNG, bas::EquivariantTensorProduct) = 
         NamedTuple() 

initialstates(rng::AbstractRNG, bas::EquivariantTensorProduct) =
         (  ranges = bas.ranges,
                LL = bas.LL, )


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
   # `ranges` and `LL` are read from the Lux state `st` (cf. initialstates) so
   # that GPU execution can use device-resident copies; `ps` is ignored.
   Lmax = maximum(st.LL)
   @assert size(Ylm, 3) >= (Lmax + 1)^2

   # one output array per L in LL. Wrapping L in a Val makes the SVector
   # length (2L+1) a compile-time constant inside _ka_evaluate_L, so each
   # kernel launch is type stable. NB: the element type of the returned tuple
   # 𝔹 still depends on the runtime values st.LL[i], so 𝔹 itself is not
   # concretely inferred (this would require LL to be a type parameter).
   𝔹 = ntuple(i -> _ka_evaluate_L(Rnl, Ylm, st.ranges[i], Val(st.LL[i])), NL)

   KernelAbstractions.synchronize(KernelAbstractions.get_backend(Rnl))
   return 𝔹
end


function _ka_evaluate_L(Rnl::AbstractArray{T, 3}, Ylm::AbstractArray{T, 3},
                        range, ::Val{L}) where {T, L}
   nneig, nnode = size(Rnl, 1), size(Rnl, 2)
   𝔹L = similar(Rnl, SVector{2*L+1, T}, (nneig, nnode, length(range)))
   backend = KernelAbstractions.get_backend(Rnl)
   kernel! = _ka_evaluate_L!(backend)
   kernel!(EquivariantTensorProduct, 𝔹L, Rnl, Ylm, range;
           ndrange = (nneig, nnode, length(range)))
   return 𝔹L
end


@kernel function _ka_evaluate_L!(::Type{EquivariantTensorProduct},
         𝔹L::AbstractArray{SVector{P, T}, 3},           # output, SVector{2L+1, T}
         @Const(Rnl::AbstractArray{T, 3}),              # (j_neig, i_node, k_feat)
         @Const(Ylm::AbstractArray{T, 3}),              # (j_neig, i_node, k_feat)
         @Const(range::AbstractVector{<:Integer})       # radial indices for this L
         ) where {P, T}
   j, i, k = @index(Global, NTuple)
   # k selects the feature, range[k] is the radial index iR into Rnl; the
   # angular part is the full L-block Ylm_{L, m}, m = -L..L, gathered below
   # (L is recovered from the SVector length 2L+1 of eltype(𝔹L)).
   iR = range[k]
   r = Rnl[j, i, iR]
   yL = _extract_yL(eltype(𝔹L), Ylm, j, i)
   𝔹L[j, i, k] = r * yL
end


# The full L-block of a SpheriCart Ylm array runs over indices lm2idx(L, -L) ..
# lm2idx(L, L), i.e. m = -L..L in order. L is recovered from the SVector length
# 2L+1, and the (l, m) -> index lookups are resolved via `lm2idx` at generation
# time, so the gather unrolls into a length-(2L+1) SVector of literal indices.
@generated function _extract_yL(::Type{SVector{P, T}}, Ylm, j, i) where {P, T}
   # P = 2 * L + 1, so L = (P - 1) ÷ 2
   L = (P - 1) ÷ 2
   vals = [ :(Ylm[j, i, $(lm2idx(L, m))]) for m = -L:L ]
   return :( SVector{P, T}( $(vals...) ) )
end

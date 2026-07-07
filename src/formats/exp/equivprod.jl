#
# implementation of an equivariant tensor product
# - should behave exactly like an equivariant linear ACE basis: 
#   input (Rnl, Ylm) => output (B0, B1, ...) 
# - 

"""
   EquivariantTensorProduct

Behaves exactly like an equivariant linear ACE basis, but without the pooling. 
Takes as input a tuple of (R, Y) edge embeddings and produces an output that 
is equivalent to an equivariant ACE basis.
"""
struct EquivariantTensorProduct{NL} # make it a Lux container layer
   prodspec::Vector{Tuple{Int, Int}}
   ranges::NTuple{NL, Vector{Int}}  # ranges[L+1] = indices of the L output in the output tuple 
   LL::NTuple{NL, Int} 
   # ---- 
   meta::Dict{String, Any}
end

function EquivariantTensorProduct(LL, Rnl_spec, Ylm_spec)
   LL = tuple(LL...)

   # 1. create the full ϕ_nlm spec (all matching products)
   #    in nametuple (n =, l = , m = ) format
   #    if the radial spec carries an `l`, then a radial (n,l) only matches an
   #    angular (l',m) when l == l'; otherwise every (n, l, m) is admissible.
   radial_has_l = (:l in fieldnames(eltype(Rnl_spec)))
   ϕ_nlm_spec = NamedTuple{(:n, :l, :m), NTuple{3, Int}}[]
   for r in Rnl_spec, y in Ylm_spec
      (radial_has_l && r.l != y.l) && continue
      push!(ϕ_nlm_spec, (n = r.n, l = y.l, m = y.m))
   end
   sort!(ϕ_nlm_spec, by = b -> (b.l, b.n, b.m))

   # 2. convert ϕ_nlm spec into a prodspec::Vector{Tuple{Int, Int}}
   #    each entry is a pair (radial index, angular index) into Rnl / Ylm
   prodspec = _make_idx_A_spec(ϕ_nlm_spec, Rnl_spec, Ylm_spec)

   # 3. select the indices of the products that match the requested LL
   #    and write them into the ranges tuple
   ranges = ntuple(i -> findall(b -> b.l == LL[i], ϕ_nlm_spec), length(LL))

   # 4. store Rnl_spec, Ylm_spec, ϕ_nlm_spec,  and LL in the meta dictionary
   meta = Dict{String, Any}("Rnl_spec" => Rnl_spec,
                            "Ylm_spec" => Ylm_spec,
                            "nlm_spec" => ϕ_nlm_spec,
                            "LL" => LL)

   return EquivariantTensorProduct(prodspec, ranges, LL, meta)
end


# ------ Claude: don't edit below here

function evaluate(op::EquivariantTensorProduct, Rnl, Ylm, ps, st)
   # ϕ is a tuple of (R, Y) features 
   # ps is a tuple of (l, l') angular momentum indices 
   # st is a tuple of (s, s') spin indices 
   error("not implemented")
end



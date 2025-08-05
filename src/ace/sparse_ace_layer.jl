using LuxCore
using Lux: glorot_normal 

struct SparseACElayer{TBB, NLL} <: AbstractLuxLayer
   symbasis::TBB    # equivariant basis -> L = 0, 1, 2, ... 
   nfeatures::NTuple{NLL, Int}   # matches symbasis.LL and symbasis.lens 
end

function Base.show(io::IO, l::SparseACElayer)
   print(io, "SparseACElayer(L = $(l.symbasis.LL), nfeat = $(l.nfeatures))")
end


_get_NLL(l::SparseACElayer) = length(l.symbasis.LL)

function LuxCore.initialparameters(rng::AbstractRNG, l::SparseACElayer)
    LL = l.symbasis.LL
    lens = l.symbasis.lens
    nfeats = l.nfeatures
    @assert length(LL) == length(nfeats) == length(lens)

    ps_symbasis = LuxCore.initialparameters(rng, l.symbasis)
    params = [ glorot_normal(rng, lens[i], nfeats[i])
               for i = 1:length(LL) ]
    ps = (symbasis = ps_symbasis, WLL = params, )
    return ps
end

LuxCore.initialstates(rng::AbstractRNG, l::SparseACElayer) = 
        (; symbasis = initialstates(rng, l.symbasis), )

(l::SparseACElayer)(X, ps, st) = evaluate(l, X, ps, st)


function evaluate(l::SparseACElayer, Φ, ps, st)
    # Φ is a tuple of embeddings. The first layer of the symbasis is the 
    # A basis (fused produce & pooling)
    B, st = ka_evaluate(l.symbasis, Φ..., ps.symbasis, st.symbasis)
    out = B .* ps.WLL 
    return out, st
end


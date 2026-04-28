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
    params = tuple( [ glorot_normal(rng, lens[i], nfeats[i])
                      for i = 1:length(LL) ]... )
    ps = (symbasis = ps_symbasis, WLL = params, )
    return ps
end

LuxCore.initialstates(rng::AbstractRNG, l::SparseACElayer) = 
        (; symbasis = initialstates(rng, l.symbasis), )

(l::SparseACElayer)(X, ps, st) = evaluate(l, X, ps, st)


function evaluate(l::SparseACElayer, Φ, ps, st)
    # Φ is a tuple of embeddings. The first layer of the symbasis is the 
    # A basis (fused produce & pooling)
    𝔹, st = ka_evaluate(l.symbasis, Φ..., ps.symbasis, st.symbasis)

    # TODO: 
    # for some reason, Zygote cannot manage the pullback through this 
    # broadcasted multiplication so we have to do it manually. 
    # Maybe using a comprehension would work and we can skip _tupmul below? 
    # out = 𝔹 .* ps.WLL 
    # out = ntuple( i -> 𝔹[i] * ps.WLL[i], length(𝔹) )
    
    out = _tupmul(𝔹, ps.WLL)


    return out, st
end

# -------------------------------------------------------------------
# temporary hack for testing purposes
#  Ai = 𝔹i, Bi = WLL[i] 
# so Ai is a matrix of svectors, B a matrix of scalar weights 
# qi = Ai * Bi  -> mat(vecs)
# <∂qi | qi> = tr(Ai * Bi * ∂qi')
# from this we can deduce the pullbacks below. 
# note that transpose(Ai) * ∂qi is mat(vec') * mat(vec) -> mat(scal)
#   which is the correct output since ∇_Bi should be mat(scal) 

function _tupmul(A, B) 
    return A .* B 
end

import ChainRulesCore: rrule 
function rrule(::typeof(_tupmul), A, B)
    out = _tupmul(A, B)
    Nt = length(A)

    @show typeof(A)
    @show typeof(B) 

    function _tupmul_pb(∂out_)
        ∂out = unthunk(∂out_)
        ∂A = ntuple(i -> ∂out[i] * transpose(B[i]), Nt)
        ∂B = ntuple(i -> transpose(A[i]) * ∂out[i], Nt)

         @show typeof(∂out)
         @show typeof(∂A)
         @show typeof(∂B)

        return (NoTangent(), ∂A, ∂B)
    end


    return out, ∂out -> _tupmul_pb(∂out) 
end


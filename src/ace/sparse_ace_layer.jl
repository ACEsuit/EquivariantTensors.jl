using LuxCore

struct sparse_ace_layer{RB, YB, BB} <: AbstractLuxLayer
   rbasis::RB      # radial embedding Rn
   ybasis::YB      # angular embedding Ylm
   symbasis::BB    # symmetric basis -> L = 0, 2
   LL::Dict{Int64, Int64}
end

function LuxCore.initialparameters(rng::AbstractRNG, m::sparse_ace_layer)
    (; (Symbol("W_$L") => [randn(rng, length(m.symbasis, L)) for _ = 1:m.LL[L]]
       for L in keys(m.LL))...)
end

LuxCore.initialstates(rng::AbstractRNG, l::sparse_ace_layer) = NamedTuple()

function (m::sparse_ace_layer)(𝐫::AbstractVector{<:SVector{3}}, ps, st)
    Rn = evaluate(m.rbasis, norm.(𝐫))
    Ylm = evaluate(m.ybasis, 𝐫)
    B = evaluate(m.symbasis, Rn, Ylm)
    out = Tuple([sum(ps[iter][i] .* B[iter]) for i in 1:length(ps[iter])] for iter in 1:length(keys(m.LL)))
    return out, st  
end

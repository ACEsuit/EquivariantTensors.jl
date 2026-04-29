
module LTM 

import Polynomials4ML as P4ML 
import EquivariantTensors as ET 
using LuxCore, LinearAlgebra, Random, Lux 
import LuxCore:  AbstractLuxLayer, initialparameters, initialstates


# radial edge embedding 
function simple_rembed(Dtot)  
   rbasis = P4ML.legendre_basis(Dtot+1)
   Rn_spec = P4ML.natural_indices(rbasis) 
   Rembed = ET.EdgeEmbed( ET.EmbedDP( 
                     ET.dp_transform( x -> 1 / (1 + norm(x.𝐫)) ), 
                     rbasis ) )
   return Rembed, Rn_spec
end

function yembed(maxl) 
   ybasis = P4ML.real_sphericalharmonics(maxl)
   Ylm_spec = P4ML.natural_indices(ybasis)
   Yembed = ET.EdgeEmbed( 
                  ET.EmbedDP( ET.dp_transform( x -> x.𝐫 ), 
                        ybasis ) )
   return Yembed, Ylm_spec
end 

function embedding(Dtot, maxl) 
   Rembed, Rn_spec = simple_rembed(Dtot)
   Yembed, Ylm_spec = yembed(maxl)
   return Parallel(nothing; Rnl = Rembed, Ylm = Yembed), Rn_spec, Ylm_spec
end

function build_basis(; Dtot, maxl, ORD, LL)
   embed, Rn_spec, Ylm_spec = embedding(Dtot, maxl)

   # generate the nnll basis pre-specification
   nnll_long = ET.sparse_nnll_set(; ORD = ORD, 
                     minn = 0, maxn = Dtot, maxl = maxl, 
                     level = bb -> sum((b.n + b.l) for b in bb; init=0), 
                     maxlevel = Dtot)

   𝔹basis = ET.sparse_equivariant_tensors(; 
                  LL = LL, mb_spec = nnll_long, 
                  Rnl_spec = Rn_spec, 
                  Ylm_spec = Ylm_spec, 
                  basis = real)

   return embed, 𝔹basis                  
end

function build_model(; Dtot, maxl, ORD, LL, NFEAT)
   embed, 𝔹basis = build_basis(; Dtot, maxl, ORD, LL)
   acel = ET.SparseACElayer(𝔹basis, NFEAT)
   return embed, acel
end


struct DotL <: AbstractLuxLayer
   nin::Int
end

function (l::DotL)(x::AbstractVector{<: Number}, ps, st)
   return dot(x, ps.W), st
end

initialparameters(rng::AbstractRNG, l::DotL) = ( W = randn(rng, l.nin), )

initialstates(rng::AbstractRNG, l::DotL) = NamedTuple()



end
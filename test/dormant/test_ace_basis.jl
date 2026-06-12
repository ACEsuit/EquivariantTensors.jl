

import Polynomials4ML as P4ML 
import EquivariantTensors as ET
using StaticArrays, LinearAlgebra, Random, Test, Zygote, LuxCore, Lux
import Optimisers as OPT
import ForwardDiff as FDiff 

@info("Preliminary Basis Jacobian test")


## 
Dtot = 8
maxl = 6
ORD = 3 
rbasis = P4ML.legendre_basis(Dtot+1)
Rn_spec = P4ML.natural_indices(rbasis) 
ybasis = P4ML.real_sphericalharmonics(maxl)
Ylm_spec = P4ML.natural_indices(ybasis)

Rembed = ET.EdgeEmbed1( 
               Chain( ET.DPTransform( (x, st) -> 1 / (1 + norm(x.𝐫)) ), 
                      rbasis ); 
               name = "Radial Edge Embedding" )

Yembed = ET.EdgeEmbed1( 
               Chain( ET.DPTransform( (x, st) -> x.𝐫 ), 
                      ybasis ); 
               name = "Angular Edge Embedding" )

# generate the nnll basis pre-specification
nnll_long = ET.sparse_nnll_set(; ORD = ORD, 
                  minn = 0, maxn = Dtot, maxl = maxl, 
                  level = bb -> sum((b.n + b.l) for b in bb; init=0), 
                  maxlevel = Dtot)

𝔹basis = ET.sparse_equivariant_tensors(; 
            LL = (0, ), mb_spec = nnll_long, 
            Rnl_spec = Rn_spec, 
            Ylm_spec = Ylm_spec, 
            basis = real )

## 

rng = Random.MersenneTwister(1234)
G = ET.Testing.rand_graph(20; nneigrg = 5:10)

psR, stR = LuxCore.setup(rng, Rembed)
R, stR = LuxCore.apply(Rembed, G, psR, stR)
(R1, ∂R), _ = ET.evaluate_ed(Rembed, G, psR, stR)
@show R1 ≈ R 
size(R) == size(R1) == size(∂R)

psY, stY = LuxCore.setup(rng, Yembed)
Y, stY = LuxCore.apply(Yembed, G, psY, stY)
(Y1, ∂Y), _ = ET.evaluate_ed(Yembed, G, psY, stY)
@show Y1 ≈ Y 
size(Y) == size(Y1) == size(∂Y)

##

ps𝔹, st𝔹 = LuxCore.setup(rng, 𝔹basis)

function eval_basis(G)
   R, _ = Rembed(G, psR, stR)
   Y, _ = Yembed(G, psY, stY)
   (B,), _ = 𝔹basis((R, Y), ps𝔹, st𝔹)
   return B 
end

# function jac_basis_fd(G) 
#    function eval_basis_R(R) 
#       new_edge_data = [ (; 𝐫 = R[i]) for i = 1:length(R) ]
#       new_G = ET.ETGraph(G.ii, G.jj, G.first, new_edge_data)
#       return eval_basis(new_G)[:]
#    end
#    R = map(x -> x.𝐫, G.edge_data)
#    # treat the many bases as a single one - this is completely ridiculous. 
#    # not sure how to even make sense of this? 
#    J1 = ForwardDiff.jacobian( eval_basis_R, R )
#    # 𝔹 = eval_basis(G) 
# end



## 

# evaluate the basis 
𝔹 = eval_basis(G) 
@show G.ii[end] == size(𝔹, 1)
@show 𝔹basis.lens[1] == size(𝔹, 2)

(R, ∂R), _ = ET.evaluate_ed(Rembed, G, psR, stR)
(Y, ∂Y), _ = ET.evaluate_ed(Yembed, G, psY, stY)
A, ∂A = ET._jacobian_X(𝔹basis.abasis, (R, Y), (∂R, ∂Y))

# ET._jacobian_X(𝔹basis, R, Y, ∂R, ∂Y)

##


model = Chain(; 
      embed = Parallel(nothing; 
               Rnl = Chain( WrappedFunction(𝐫 -> norm.(𝐫)),  
                            rbasis ), 
               Ylm = ybasis),
      𝔹 = 𝔹basis, 
      y01 = Parallel(nothing; 
            y0 = DotL(length(𝔹basis, 0)), 
            y1 = DotL(length(𝔹basis, 1)), ), 
      out = WrappedFunction(x -> x[1] + sum(abs2, x[2]) )
      )

##

__rand_sphere() = ( u = randn(SVector{3, Float64}); u / norm(u) )
__rand_x() = (0.1 + 0.9 * rand()) * __rand_sphere()
nX = 7
𝐫 = [ __rand_x() for _ = 1:nX ]

rng = Random.MersenneTwister(1234)
ps, st = Lux.setup(rng, model)
φ, _ = Lux.apply(model, 𝐫, ps, st)

## 

# differentiate with ForwardDiff
pvec, _rest = OPT.destructure(ps)
gf = _rest(FDiff.gradient(p -> Lux.apply(model, 𝐫, _rest(p), st)[1], pvec))

# Differentiate with Zygote
gz, = Zygote.gradient(p -> Lux.apply(model, 𝐫, p, st)[1], ps)

println(@test OPT.destructure(gf)[1] ≈ OPT.destructure(gz)[1])


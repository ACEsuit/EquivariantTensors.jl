
using LinearAlgebra, Lux, Random, EquivariantTensors, Test, Zygote, 
      ForwardDiff, StaticArrays
using ACEbase.Testing: print_tf, println_slim

using EquivariantTensors: SelectLinL 

import EquivariantTensors as ET 
import Polynomials4ML as P4ML      
import KernelAbstractions as KA

rng = MersenneTwister(1234)

@info("--------------------------------------------------------- ")
@info("Testing various transformed embedding layers ")
@info("--------------------------------------------------------- ")

##
#
@info("TEST 1: test a transformed radial basis with scalar number as input ")


npoly = 15
nmax = 7
rtrans = r -> 1 ./ (1 .+ r)
polybasis = P4ML.ChebBasis(npoly)
length(polybasis) == npoly || error("mismatch assumed basis length")
lin = P4ML.Utils.LinL(npoly, nmax) 

rbasis = Chain(; trans = WrappedFunction(rtrans), basis = polybasis, lin = lin)
ps_r, st_r = LuxCore.setup(rng, rbasis)

# batch evaluation, differentiation 
rr = rand(100)
Rn, _ = rbasis(rr, ps_r, st_r)  
U = randn(size(Rn)) 

f = _rr -> dot(U, rbasis(_rr, ps_r, st_r)[1])
g = Zygote.gradient(f, rr)[1]
println_slim(@test ForwardDiff.gradient(f, rr) ≈ g)

##
#
@info("TEST 2: test a transformed radial basis with NamedTuple as input")

rtrans_nt = ET.NTtransform(x -> 1 / (1 + norm(x.𝐫)))
xx = [ (𝐫 = (@SVector rand(3))/sqrt(3), z = rand(1:3)) for _=1:100 ]
rr = [ norm(x.𝐫) for x in xx ]

rbasis_nt = Chain(; trans = rtrans_nt, basis = polybasis, lin = lin)
ps_rnt, st_rnt = LuxCore.setup(rng, rbasis_nt)
ps_rnt.lin.W[:] .= ps_r.lin.W[:]  # make sure the linear part is the same

Rn1, _ = rbasis(rr, ps_r, st_r)
Rn2, _ = rbasis_nt(xx, ps_rnt, st_rnt)
println_slim(@test Rn1 ≈ Rn2)

f2 = _xx -> dot(U, rbasis_nt(_xx, ps_rnt, st_rnt)[1])
g2 = Zygote.gradient(f2, xx)[1] 
g2_𝐫 = [ dx.𝐫 for dx in g2 ]

# check this: 
g1_r = Zygote.gradient(f, rr)[1]
g1_𝐫 = g1_r .* [ x.𝐫 / norm(x.𝐫) for x in xx ]
println_slim(@test g1_𝐫 ≈ g2_𝐫)

## 
#
@info("TEST 3: test an angular embedding ")

maxl = 4
sh = P4ML.real_sphericalharmonics(maxl; T = Float64, static=true)
ybasis = Chain(; trans = ET.NTtransform(x -> x.𝐫), 
                 basis = sh )
ps_y, st_y = LuxCore.setup(rng, ybasis)
𝐫𝐫 = [ x.𝐫 for x in xx ]
Y1, dY1 = P4ML.evaluate_ed(sh, 𝐫𝐫)

Y2 = ybasis(xx, ps_y, st_y)[1]
println_slim(@test Y1 ≈ Y2)

V = randn(size(Y1))
fy = _xx -> dot(V, ybasis(_xx, ps_y, st_y)[1])
gy = Zygote.gradient(fy, xx)[1]
gy_𝐫 = [ dx.𝐫 for dx in gy ]
gY1 = sum(V .* dY1, dims=2)[:]
println_slim(@test gy_𝐫 ≈ gY1)

##
#
@info("TEST 4: test a parallel embedding with both radial and angular parts")

embed = BranchLayer(Rnl = rbasis_nt, Ylm = ybasis)
ps, st = LuxCore.setup(rng, embed)
ps.Rnl.lin.W[:] .= ps_r.lin.W[:]

# check this evaluates the right thing
(Rn4, Ylm4), _ = embed(xx, ps, st)
println_slim(@test Rn4 ≈ Rn2)
println_slim(@test Ylm4 ≈ Y2)

# check this differentiates correctly. 
fe = _xx -> begin 
      (R, Y) = embed(_xx, ps, st)[1]
      return dot(R, U) + dot(Y, V)
   end 
ge = Zygote.gradient(fe, xx)[1]
ge_𝐫 = [ dx.𝐫 for dx in ge ]

println_slim(@test ge_𝐫 ≈ gy_𝐫 + g2_𝐫)

##
#
@info("TEST 5: build a more complicated radial embedding ")

maxpoly = 15; maxn = 7; maxz = 3;
rbasis = let maxpoly = maxpoly, maxn = maxn, maxz = maxz, aa = 0.5 .+ 0.5 * rand(maxz)

   # x --> y = y(x.r, x.z)     ... transformed coordinates 
   get_radial = ET.NTtransform(x -> begin 
            z = x.z 
            a = aa[z]
            r = norm(x.𝐫) 
         return 1 / (1 + a * r^2 )
      end )

   # y --> P(y)   ... polynomial basis in transformed coordinates 
   polys = P4ML.ChebBasis(maxpoly)
   
   # P --> W[x.z] * P     ... make basis learnable with z-dependent weights 
   lin = SelectLinL(maxpoly, maxn, maxz, x -> x.z)

   # put it all together 
   SkipConnection( Chain(; rtrans = get_radial, polys = polys), 
                   lin )
end

ps, st = LuxCore.setup(rng, rbasis)

R, _ = rbasis(xx, ps, st)

# check correctness by evaluating this "manually" 
get_y = rbasis.layers.layers.rtrans.f
_yy = get_y.(xx)
_P = (P4ML.ChebBasis(maxpoly))(_yy)
_R = reduce(vcat, (ps.connection.W[:, :, x.z] * _P[i, :])' 
            for (i, x) in enumerate(xx) )
println_slim(@test _R ≈ R) 

# check differentiation
U = randn(size(R))
fr = _xx -> dot(U, rbasis(_xx, ps, st)[1])
gr = Zygote.gradient(fr, xx)[1]
gr_𝐫 = [ dx.𝐫 for dx in gr ]

𝐫2xx(𝐫𝐫) = [ (𝐫 = 𝐫, z = x.z) for (𝐫, x) in zip(𝐫𝐫, xx) ] 
𝐫𝐫 = [ x.𝐫 for x in xx ]

using ACEbase 
success = ACEbase.Testing.fdtest(_𝐫𝐫 -> fr(𝐫2xx(_𝐫𝐫)), _𝐫𝐫 -> gr_𝐫, 𝐫𝐫)
println_slim(@test success)

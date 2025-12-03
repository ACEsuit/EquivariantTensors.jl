using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ".."))
using TestEnv; TestEnv.activate();
Pkg.develop(url = joinpath(@__DIR__(), "..", ".."))

##

using Test 

import EquivariantTensors as ET 
using StaticArrays, LinearAlgebra, LuxCore, Random, ForwardDiff
using ACEbase.Testing: print_tf
rng = MersenneTwister(1234)

##

function make_rand_agnesi() 
   pcut = rand(2:4)
   pin = rand(pcut:6)

   cats = SA['a', 'b', 'c']

   rins = Dict(); reqs = Dict(); rcuts = Dict();
   for i = 1:length(cats), j = i:length(cats)
      c1 = cats[i]; c2 = cats[j]
      rins[(c2, c1)] = rins[(c1, c2)] = 0.33 * rand()
      reqs[(c2, c1)] = reqs[(c1, c2)] = 1.0 + 0.4*rand()
      rcuts[(c2, c1)] = rcuts[(c1, c2)] = 2.5 + rand()
   end

   trans = ET.agnesi_transform(cats, rins, reqs, rcuts, pin, pcut)
   return trans, cats, pcut, pin, rins, reqs, rcuts
end 

rand_sphere() = (u = (@SVector randn(3)); u / norm(u))

rand_x(cats, maxrcut) = (𝐫 = (1.1 * maxrcut) * rand() * rand_sphere(), 
               s0 = rand(cats), s1 = rand(cats)) 

function _test_agnesi(cats, pcut, pin, rins, reqs, rcuts)

   maxrcut = maximum(collect(values(rcuts)))   
   x = rand_x(cats, maxrcut) 
   y = trans(x, ps, st)[1] 

   # check against manual implementation 
   p = ET.agnesi_params(pcut, pin, rins[(x.s0, x.s1)], 
                            reqs[(x.s0, x.s1)], rcuts[(x.s0, x.s1)])
   r = norm(x.𝐫)                            
   y1 = ET.eval_agnesi(r, p)

   __s(r) = (r - rins[(x.s0, x.s1)]) / (reqs[(x.s0, x.s1)] - rins[(x.s0, x.s1)])
   __x(s) = 1 / (1 + p.a * s^(pin) / (1 + s^(pin - pcut)))
   _xin = __x(__s(rins[(x.s0, x.s1)]))
   _xcut = __x(__s(rcuts[(x.s0, x.s1)]))
   b1 = 2 / (_xin - _xcut)
   b0 = -1 - 2 * _xcut / (_xin - _xcut)
   y2 = b1 * __x(__s(r)) + b0
   y2 = max(-1, min(1, y2))

   return y, y1, y2                  
end

##

@info("Testing agnesi transform consistency")

for ntest = 1:10 
   trans, cats, pcut, pin, rins, reqs, rcuts = make_rand_agnesi()
   ps, st = LuxCore.setup(rng, trans)
   

   for ntest2 = 1:10 
      y, y1, y2 = _test_agnesi(cats, pcut, pin, rins, reqs, rcuts)
      print_tf(@test y ≈ y1 ≈ y2) 
   end
   print("|")
end 
println() 

##

@info("Testing that the slope is maximal at req")
for ntest = 1:30
   trans, cats, pcut, pin, rins, reqs, rcuts = make_rand_agnesi()
   ps, st = LuxCore.setup(rng, trans)
   maxrcut = maximum(collect(values(rcuts)))   

   x = rand_x(cats, maxrcut)

   params = ET.agnesi_params(pcut, pin, rins[(x.s0, x.s1)], 
                           reqs[(x.s0, x.s1)], rcuts[(x.s0, x.s1)])
   rin = rins[(x.s0, x.s1)]                         
   req = reqs[(x.s0, x.s1)]
   rcut = rcuts[(x.s0, x.s1)]

   dy(r) = ForwardDiff.derivative(r -> ET.eval_agnesi(r, params), r)
   dy_eq = dy(req)

   rr = rin .+ (rcut - rin) * rand(100)
   dy_rr = dy.(rr)
   print_tf(@test all(dy_rr .>= dy_eq))
end


## 

@info("Test agnesi transform for atomic species")
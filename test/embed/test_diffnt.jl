
using EquivariantTensors, StaticArrays, Test, ForwardDiff
using ACEbase.Testing: println_slim, print_tf 

import EquivariantTensors as ET
import ACEbase: evaluate 

##

@info("Testing DiffNT")
@info(" ... preliminaries")

rand_x() = (q = randn(), r = randn(SVector{3, Float64}), z = rand(1:10))

x = rand_x()
v_nt = ET.DiffNT._ctsnt(x) 
println_slim(@test v_nt == (q = x.q, r = x.r))
v = ET.DiffNT._nt2svec(v_nt)
println_slim(@test v == SVector{4, Float64}(x.q, x.r...))
v_nt1 = ET.DiffNT._svec2nt(v, v_nt)
println_slim(@test v_nt1 == v_nt)

##

@info("Testing the differentiation interface") 

module Test_DiffNT 
   using StaticArrays, ForwardDiff 
   import EquivariantTensors as ET
   import EquivariantTensors: evaluate 

   struct F{N, T}; W::SVector{N, T}; end

   # random expression, but representative in terms of simplicity 
   ET.evaluate(f::F, x) = sum(x.r .* x.r) * x.q / (1 + f.W[x.z]^2)   
   (f::F)(x::NamedTuple) = evaluate(f, x)

   # manual gradient 
   grad_man(f::F, x) = ( r2 = sum(x.r .* x.r); w = 1 / (1 + f.W[x.z]^2); 
                        (q = r2 * w, r = 2 * x.r * x.q * w, )  )

   # gradient via ForwardDiff                       
   function grad_1(f::F, x) 
      ∂q = ForwardDiff.derivative(q -> evaluate(f, (q=q,   r=x.r, z=x.z)), x.q)
      ∂r = ForwardDiff.gradient(r -> evaluate(f, (q=x.q, r=r,   z=x.z)), x.r)
      return (q = ∂q, r = ∂r)
   end

   grad_2(f::F, x) = ET.DiffNT.grad_fd(f, x)
end 

## 

f = Test_DiffNT.F(@SVector randn(10))

for ntest = 1:20 
   local x 
   x = rand_x() 
   g_man = Test_DiffNT.grad_man(f, x) 
   g_fd1 = Test_DiffNT.grad_1(f, x)
   g_fd2 = Test_DiffNT.grad_2(f, x)
   print_tf(@test ( all(g_fd1[sym] ≈ g_man[sym] for sym in fieldnames(g_man)) ))
   print_tf(@test ( all(g_fd2[sym] ≈ g_man[sym] for sym in fieldnames(g_man)) ))
end 
println() 

##

@info("allocation tests")

N = 1000 
X = [ rand_x() for _ in 1:N ]
TG = typeof(Test_DiffNT.grad_2(f, X[1]))
gY = Vector{TG}(undef, N)

function count_alloc(f, Y, X) 
   gfun = x -> Test_DiffNT.grad_2(f, x)
   map!(gfun, gY, X)
   @allocations map!(gfun, gY, X)
end

println_slim(@test count_alloc(f, gY, X) <= 1) 

##
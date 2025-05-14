
import EquivariantTensors: O3
using Polynomials4ML, StaticArrays, BlockDiagonals
using Test

isdefined(Main, :___UTILS_FOR_TESTS___) || include("../utils/utils_testO3.jl")

## 

@info("Test O3 Yvector -> Ymatrix transformation")
for L1 = 0:2, L2 = 0:2, basis in (real, complex) 
   # print("L1 = $L1, L2 = $L2, basis = $basis : ")
   L = L1 + L2 
   trans = O3.TYVec2YMat(L1, L2; basis = basis)
   ybasis = ( basis == real ? real_sphericalharmonics(L)
                            : complex_sphericalharmonics(L) )
   𝐫 = @SVector randn(3)
   θ = π * rand(3)
   Q = O3.Q_from_angles(θ)
   DL1 = O3.D_from_angles(L1, θ, basis)
   DL2 = O3.D_from_angles(L2, θ, basis)
   y = O3.SYYVector(ybasis(𝐫))
   hy = trans(y)
   yQ = O3.SYYVector(ybasis(Q*𝐫))

   Dall = BlockDiagonal([ O3.D_from_angles(l, θ, basis) for l = 0:L ])
   print_tf(@test (yQ ≈ Dall * y))

   hyQ = trans(yQ)
   print_tf(@test (hyQ ≈ DL1 * hy * DL2' ≈ trans(O3.SYYVector(Dall * y))))
   # println() 
end


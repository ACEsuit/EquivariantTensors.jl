
import EquivariantTensors: O3
using Polynomials4ML, StaticArrays, BlockDiagonals
using Test

isdefined(Main, :___UTILS_FOR_TESTS___) || include("../utils/utils_testO3.jl")

## 

L1 = 1; L2 = 1 
L = L1 + L2 
trans = O3.TYVec2YMat(L1, L2; basis = real)
trans_old = O3.TYVec2pp(; basis = real)
ybasis = real_sphericalharmonics(L)
𝐫 = @SVector randn(3)
θ = π * rand(3)
Q, DL = O3.QD_from_angles(L, θ, real)
DL1 = O3.D_from_angles(L1, θ, real)
DL2 = O3.D_from_angles(L2, θ, real)
y = O3.SYYVector(ybasis(𝐫))
hy = trans(y)
yQ = O3.SYYVector(ybasis(Q*𝐫))

Dall = BlockDiagonal([ O3.D_from_angles(l, θ, real) for l = 0:L ])
@show yQ ≈ Dall * y 

##

hyQ_old = trans_old(yQ[1:1], yQ[5:9])
hyQ = trans(yQ)
@show hyQ ≈ DL1 * hy * DL2' ≈ trans(O3.SYYVector(Dall * y))

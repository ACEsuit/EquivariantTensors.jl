
using EquivariantTensors.O3: coupling_coeffs

nn = [1,2,3]
ll = [2, 2, 2]
L = 1 
cUrpe, cMll_rpe = coupling_coeffs(L, ll, nn)
rUrpe, rMll_rpe = coupling_coeffs(L, ll, nn; basis=real)
@show size(cUrpe), size(rUrpe)

using EquivariantTensors, StaticArrays, LinearAlgebra
using EquivariantTensors.O3: coupling_coeffs

"""
D(N,l,Ltot)
Compute the dimension of the RPE basis for a given set of parameters
N: Correlation order
l: Angular momentum
Ltot: Order of equivariance
"""
function D(N,l,Ltot)
    ll = SA[ones(Int64,N)...] .* l
    nn = SA[ones(Int64,N)...]
    N1 = Int(round(N/2))

    # t_rpe = @elapsed C_rpe, M = coupling_coeffs(Ltot,ll,nn) # reference time
    try 
        global t_rpe_recursive_kernel = @elapsed global C_rpe_recursive, MM = coupling_coeffs(Ltot,ll,nn,N1)
    catch
        global t_rpe_recursive_kernel = @elapsed global C_rpe_recursive, MM = coupling_coeffs(Ltot,ll,nn)
    end
    
    println("Case : nn = $nn, ll = $ll, Ltot = $Ltot, N1 = $N1")
    # println("Standard RPE basis : $t_rpe")
    println("Time computing the RPE basis recursively: $t_rpe_recursive_kernel" * "(s)")
    println("Dimensionality : $(size(C_rpe_recursive,1))")
    
    return size(C_rpe_recursive,1)
end

[ D(N,2,0) for N = 1:15 ]

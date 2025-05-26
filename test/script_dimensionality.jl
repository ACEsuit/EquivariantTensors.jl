using EquivariantTensors, StaticArrays, LinearAlgebra
using EquivariantTensors.O3: coupling_coeffs, gram

"""
_D(N,l,Ltot)
Compute the dimension of the RPE basis for a given set of parameters
N: Correlation order
l: Angular momentum
Ltot: Order of equivariance
"""
function _D(N::Int,l::Int,Ltot::Int,N1 = Int(round(N/2)))
    ll = SA[ones(Int64,N)...] .* l
    nn = SA[ones(Int64,N)...]
    
    # t_rpe = @elapsed C_rpe, M = coupling_coeffs(Ltot,ll,nn) # reference time
    try 
        global t_rpe_recursive_kernel = @elapsed global C_rpe_recursive, MM = coupling_coeffs(Ltot,ll,nn,N1)
    catch
        global t_rpe_recursive_kernel = @elapsed global C_rpe_recursive, MM = coupling_coeffs(Ltot,ll,nn)
    end
    
    println("Case : nn = $nn, ll = $ll, Ltot = $Ltot, N1 = $N1")
    # println("Standard RPE basis : $t_rpe")
    println("Time computing the RPE basis recursively: $t_rpe_recursive_kernel" * "(s)")
    println("Dimensionality : $(findall(x -> x>1e-7, norm.([ C_rpe_recursive[i,:] for i in 1:size(C_rpe_recursive,1) ]) ./ size(C_rpe_recursive,2)) |> length)")
    
    # For large N, our code can still return zero basis even though the non-recursive version returns already a normalized basis
    return findall(x -> x>1e-7, norm.([ C_rpe_recursive[i,:] for i in 1:size(C_rpe_recursive,1) ]) ./ size(C_rpe_recursive,2)) |> length
end


const D_cache = Dict{Tuple{Int, Int, Int}, Int}()

# Cached wrapper for _D
"""
D(N,l,Ltot)
Compute the dimension of the RPE basis for a given set of parameters
N: Correlation order
l: Angular momentum
Ltot: Order of equivariance
"""
function D(N::Int,l::Int,Ltot::Int)
    key = (N,l,Ltot)
    if haskey(D_cache, key)
        return D_cache[key]
    else
        result = _D(N,l,Ltot)
        D_cache[key] = result
        return result
    end
end

# Functions that find (N,l) and (N',l') such that D(N,l,L) == D(N',l',L) for all L (and Nl == N'l')
function factor_pairs(N::Int)
    @assert N > 0 "N must be positive"
    pairs = Tuple{Int, Int}[]
    for a in 1:floor(Int, sqrt(N))
        if N % a == 0
            b = div(N, a)
            push!(pairs, (a, b))
            if a != b
                push!(pairs, (b, a))  # Include symmetric pair
            end
        end
    end
    return pairs
end

DD(aa) = [ D(aa[1],aa[2],L) for L = 0:aa[1]*aa[2] ]

function matching_pairs(AA::Vector{Tuple{Int, Int}}, B::Function)
    groups = Dict{Any, Vector{Tuple{Int, Int}}}()

    # Group all pairs by B(pair)
    for aa in AA
        key = B(aa)
        push!(get!(groups, key, Tuple{Int, Int}[]), aa)
    end

    # Collect all unordered pairs from each group
    result = Tuple{Tuple{Int, Int}, Tuple{Int, Int}}[]
    for group in values(groups)
        n = length(group)
        for i in 1:n-1, j in i+1:n
            push!(result, (group[i], group[j]))
        end
    end

    return result
end

# Example code for finding matching pairs for N*l=4
# matching_pairs(factor_pairs(4), DD)

# Example code for saving cached dimensions to a JLD2 file
# using JLD2

# for N in 1:6, l in 1:6
#     for Ltot in 0:N*l
#         D(N,l,Ltot)
#     end
# end

# filename = "cache_dim.jld2"
# @save filename D_cache
# println("Saved cache to $filename")

# # Load cache from a .jld2 file
# @load filename D_cache
# println("Loaded cache from $filename")
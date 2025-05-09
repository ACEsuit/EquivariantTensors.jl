module O3new
 
using PartialWaveFunctions
using Combinatorics
using LinearAlgebra
using StaticArrays
using SparseArrays

export coupling_coeffs

include("O3_utils.jl")
include("yyvector.jl")
include("O3_transformations.jl")



# -----------------------------------------------------

# The generalized Clebsch Gordan Coefficients; variables of this function are 
# fully inherited from the first ACE paper. 
function GCG(l::SVector{N,Int64}, m::SVector{N,Int64}, L::SVector{N,Int64},
             M_N::Int64; flag=:cSH) where N
    # @assert -L[N] ≤ M_N ≤ L[N] 
    if mm_filter_single(m, M_N;flag=flag) == false || L[1] < abs(m[1])
        return 0.
    end

    if flag == :cSH
        M = m[1]
        C = 1.
        for k in 2:N
            if L[k] < abs(M+m[k])
                return 0.
            else
                C *= PartialWaveFunctions.clebschgordan(
                            L[k-1], M, l[k], m[k], L[k], M+m[k])
                M += m[k]
            end
        end
        return C
    else
        C = 0.
        for M in signed_m(M_N)
            ext_mset = filter( x -> sum(x) == M, signed_mmset(m) )
        
            for mm in ext_mset
                mm = SA[mm...]
                @assert sum(mm) == M
                C_loc = GCG(l,mm,L,M;flag=:cSH)
                coeff = Ctran(M_N,M;convention=flag)' * 
                           prod( Ctran(m[i],mm[i];convention=flag) for i in 1:N )
                C_loc *= coeff
                C += C_loc
            end
        end

        # We actually expect real values 
        return abs(C - real(C)) < 1e-12 ? real(C) : C 
    end

end

# Only when M_N = sum(m) can the CG coefficient be non-zero, so when missing M_N, 
# we return either 
# (1) the full CG coefficient given l, m and L, as a rank 1 vector; 
# (2) or the only one element that can possibly be non-zero on the above vector.
# I suspect that the first option will not be used anyhow, but I keep it for now.
function GCG(l::SVector{N,Int64}, m::SVector{N,Int64}, L::SVector{N,Int64};
             vectorize::Bool=true, flag=:cSH) where N 
    if flag == :cSH
        return (vectorize ? (GCG(l,m,L,sum(m);flag=flag) * 
                                Float64.(I(2L[N]+1)[sum(m)+L[N]+1,:])) 
                          : GCG(l,m,L,sum(m);flag=flag) )
    else
        if vectorize == false && L[N] != 0
            error("""For the rSH basis, the CG coefficient is always a vector 
                     except for the case of L=0.""")
        else
            return (L[N] == 0 ? GCG(l,m,L,L[N];flag=flag) 
                              : SA[[ GCG(l,m,L,M_N;flag=flag) 
                                     for M_N in -L[N]:L[N] ]...]  )
        end
    end
end

# Function that returns a L set given an `l`. The elements of the set start with 
# l[1] and end with L. 
function SetLl(l::SVector{N,Int64}, L::Int64) where N
    T = typeof(l)
    if N==1
        return l[1] == L ? [T(l[1])] : Vector{T}[]
    elseif N==2        
        return abs(l[1]-l[2]) ≤ L ≤ l[1] + l[2] ? [T(l[1],L)] : Vector{T}[]
    end
    
    set = [ [l[1];] ]
    for k in 2:N
        set_tmp = set
        set = Vector{Any}[]
        for a in set_tmp
            if k < N
                for b in abs(a[k-1]-l[k]):a[k-1]+l[k]
                    push!(set, [a; b])
                end
            elseif k == N
                if (abs.(a[N-1]-l[N]) <= L)&&(L <= (a[N-1]+l[N]))
                    push!(set, [a; L])
                end
            end
        end
    end  

    return T.(set)
end

SetLl(l::SVector{N,Int64}) where N = union([SetLl(l, L) for L in 0:sum(l)]...)

function Sn(nn,ll)
    # should assert that lexicographical order
    N = length(ll)
    @assert length(ll) == length(nn)
    perm_indices = [1]
    for i in 2:N
        if ll[i] != ll[perm_indices[end]] || nn[i] != nn[perm_indices[end]]
            push!(perm_indices,i)
        end
    end
    return [perm_indices;N+1]
end

# The set of integers that has the same absolute value as m
signed_m(m) = unique([m,-m]) 

# The set of vectors whose i-th element has the same absolute value as m[i] 
# for all i
signed_mmset(m) = Iterators.product([signed_m(m[i]) for i in 1:length(m)]...
                                   ) |> collect 

function mm_filter_single(mm::Union{Vector{Int64},SVector{N,Int64}}, k::Int64; 
                 flag=:cSH) where N
    if flag == :cSH
        return sum(mm) == k
    else
        # for the rSH, the criterion is that whether there exists a combinition 
        # of [+/- m_i]_i, such that the sum of the combination equals to k
        return any([sum(mm1) == k for mm1 in signed_mmset(mm)])
    end
end

function mm_filter(mm::Union{Vector{Int64},SVector{N,Int64}}, L::Int64; 
                 flag=:cSH) where N
    if flag == :cSH
        return abs(sum(mm)) <= L
    else
        # for the rSH, the criterion is that whether there exists a combinition 
        # of [+/- m_i]_i, such that the sum of the combination equals to k
        return any([abs.(sum(mm1)) <= L for mm1 in signed_mmset(mm)])
    end
end

# Function that generates the set of ordered m's given `n` and `l` with 
# the absolute sum of  m's smaller than or equal to `L`.
#
# NB: This function assumes lexicographical ordering

function mm_generate(L::Int, ll::T, nn::T; 
                     # PI = !(isnothing(nn)), 
                     flag = :cSH) where {T} 
    N = length(ll)
    @assert length(ll) == length(nn)
    # S = Sn(nn,ll)
    ci = CartesianIndices(ntuple(t -> -ll[t]:ll[t], N))
    MM = Vector{T}(undef, length(ci))
    for (i, I) in enumerate(ci)
        MM[i] = I.I 
    end 

    # No matter PI or not, this fcn always generates all admissible mm's
    # and if PI, they are just filtered in _coupling_coeffs
    _mm_filter = x -> mm_filter(x, L; flag)
    
    return MM[findall(x -> x==1, _mm_filter.(MM))]
end

function gram(X::Matrix{SVector{N,T}}) where {N,T}
    G = zeros(T, size(X,1), size(X,1))
    for i = 1:size(X,1)
       for j = i:size(X,1)
          G[i,j] = sum(dot(X[i,t], X[j,t]) for t = 1:size(X,2))
          i == j ? nothing : (G[j,i]=G[i,j]')
       end
    end
    return G
 end

gram(X::Matrix{<:Number}) = X * X'

function lexi_ord(nn, ll)
   N = length(nn)
   bb = [ (ll[i], nn[i]) for i = 1:N ]
   p = sortperm(bb)
   bb_sorted = bb[p]
   return SVector{N, Int}(ntuple(i -> bb_sorted[i][2], N)), 
          SVector{N, Int}(ntuple(i -> bb_sorted[i][1], N)), 
          invperm(p)
end

"""
    O3.coupling_coeffs(L, ll, nn; PI, basis)
    O3.coupling_coeffs(L, ll; PI, basis)

Compute coupling coefficients for the spherical harmonics basis, where 
- `L` must be an `Integer`;
- `ll, nn` must be vectors or tuples of `Integer` of the same length.
- `PI`: whether or not the coupled basis is permutation-invariant (or the 
corresponding tensor symmetric); default is `true` when `nn` is provided 
and `false` when `nn` is not provided.
- `basis`: which basis is being coupled, default is `complex`, alternative
choice is `real`, which is compatible with the `SpheriCart.jl` convention.  
"""
function coupling_coeffs(L::Integer, ll, nn = nothing; 
                         PI = !(isnothing(nn)), 
                         basis = complex)

    # convert L into the format required internally 
    _L = Int(L) 

    # convert ll into an SVector{N, Int}, as required internally 
    N = length(ll) 
    _ll = try 
        _ll = SVector{N, Int}(ll...)
    catch 
        error("""coupling_coeffs(L::Integer, ll, ...) requires ll to be 
               a vector or tuple of integers""")
    end

    # convert nn into an SVector{N, Int}, as required internally 
    if isnothing(nn) 
        if PI 
            _nn = SVector{N, Int}(ntuple(i -> 0, N)...)
        else 
            _nn = SVector{N, Int}((1:N)...)
        end
    elseif length(nn) != N 
        error("""coupling_coeffs(L::Integer, ll, nn) requires ll and nn to be 
               of the same length""")
    else
        _nn = try 
            _nn = SVector{N, Int}(nn...)
        catch 
            error("""coupling_coeffs(L::Integer, ll, nn) requires nn to be 
                   a vector or tuple of integers""")
        end
    end 

    if basis == complex 
        flag = :cSH 
    elseif basis == real 
        flag = :SpheriCart
    elseif basis isa Symbol
        flag = basis 
    else 
        error("unknown basis type: $basis")
    end
    
    return _coupling_coeffs(_L, _ll, _nn; PI = PI, flag = flag)
end

function _sort(x::T, permutable_blocks::Vector{Vector{Int}}) where T
    # Sorts the vector x according to the indices in permutable_blocks
    # This is used to sort the equivalent classes of m's
    x = Vector{eltype(x)}(x)
    for block in permutable_blocks
        x[block] = sort(x[block])
    end
    return T(x)
end


# Function that generates the coupling coefficient of the RE basis (PI = false) 
# or RPE basis (PI = true) given `nn` and `ll`. 
function _coupling_coeffs(L::Int, ll::SVector{N, Int}, nn::SVector{N, Int}; 
                          PI = true, flag = :cSH) where N

    # NOTE: because of the use of m_generate, the input (nn, ll ) is required
    # to be in lexicographical order.
    nn, ll, inv_perm = lexi_ord(nn, ll)

    Lset = SetLl(ll,L)
    r = length(Lset)
    T = L == 0 ? Float64 : SVector{2L+1,Float64}
    if r == 0; return zeros(T, 0, 0), SVector{N, Int}[]; end
     
    if !PI
        MM = mm_generate(L, ll, nn; flag=flag) # all m's
        UMatrix = zeros(T, r, length(MM)) # Matrix containing the coupling coefs D
        for i in 1:r
            for (j,mm) in enumerate(MM)
                UMatrix[i,j] = GCG(ll,mm,Lset[i];vectorize=(L!=0),flag=flag)
            end
        end 
        return UMatrix, [mm[inv_perm] for mm in MM]
    else
        # permutation blocks - within which the nn and ll are identical
        S = Sn(nn,ll)
        permutable_blocks = [ Vector([S[i]:S[i+1]-1]...) for i in 1:length(S)-1]

        MM = mm_generate(L, ll, nn; flag=flag) # all admissible mm's
        MM_sorted = [ _sort(mm, permutable_blocks) for mm in MM ] # sort the mm's within the permutable blocks
        MM_reduced = unique(MM_sorted) # ordered mm's - representatives of the equivalent classes
        D_MM_reduced = Dict(MM_reduced[i] => i for i in 1:length(MM_reduced))
        
        FMatrix=zeros(T, r, length(MM_reduced)) # Matrix containing f(m,i)

        for (j,mm) in enumerate(MM)
            col = D_MM_reduced[MM_sorted[j]] # avoid looking up the dictionary repeatedly
            for i in 1:r
                FMatrix[i,col] += GCG(ll,mm,Lset[i];vectorize=(L!=0),flag=flag)
            end
        end 
        
        # Linear dependence
        U, S, V = svd(gram(FMatrix))
        # Somehow rank is not working properly here, might be a relative  
        # tolerance issue.
        # original code: rank(Diagonal(S); rtol =  1e-12) 
        rk = findall(x -> x > 1e-12, S) |> length 
        # return the RE-PI coupling coeffs
        return Diagonal(sqrt.(S[1:rk])) * U[:, 1:rk]' * FMatrix, 
               [ mm[inv_perm] for mm in MM_reduced ]
    end
end

end
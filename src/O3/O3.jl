module O3
 
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
function GCG(ll::SVector{N,Int64}, mm::SVector{N,Int64}, LL::SVector{N,Int64},
             M_N::Int64, basis::typeof(complex)) where {N}
    # @assert -L[N] ≤ M_N ≤ L[N] 
    if mm_filter_single(mm, M_N, basis) == false || LL[1] < abs(mm[1])
        return 0.0
    end

    M = mm[1]
    C = 1.
    for k in 2:N
        if LL[k] < abs(M+mm[k])
            return 0.
        else
            C *= PartialWaveFunctions.clebschgordan(
                        LL[k-1], M, ll[k], mm[k], LL[k], M+mm[k])
            M += mm[k]
        end
    end
    return C
end 

function GCG(ll::SVector{N,Int64}, mm::SVector{N,Int64}, LL::SVector{N,Int64},
             M_N::Int64, basis::typeof(real)) where {N}
    # @assert -L[N] ≤ M_N ≤ L[N] 
    if mm_filter_single(mm, M_N, basis) == false || LL[1] < abs(mm[1])
        return 0.0
    end

    C = 0.0
    for M in signed_m(M_N)
        ext_mset = filter( x -> sum(x) == M, signed_mmset(mm) )
    
        for __mm in ext_mset
            _mm = SA[__mm...]
            @assert sum(_mm) == M
            C_loc = GCG(ll, _mm, LL, M, basis)
            coeff = _Ctran(M_N, M, basis)' * 
                        prod( _Ctran(mm[i], _mm[i], basis) for i in 1:N )
            C_loc *= coeff
            C += C_loc
        end
    end

    @assert abs(C - real(C)) < 1e-12
    return real(C) 
end

# Only when M_N = sum(m) can the CG coefficient be non-zero, so when missing M_N, 
# we return either 
# (1) the full CG coefficient given l, m and L, as a rank 1 vector; 
# (2) or the only one element that can possibly be non-zero on the above vector.
# I suspect that the first option will not be used anyhow, but I keep it for now.
function GCG(ll::SVector{N,Int64}, mm::SVector{N,Int64}, LL::SVector{N,Int64};
             vectorize::Bool=true, basis = complex) where N 
    if basis === complex
        return (vectorize ? (GCG(ll, mm, LL, sum(mm), basis) * 
                                Float64.(I(2*LL[N]+1)[sum(mm)+LL[N]+1,:])) 
                          : GCG(ll, mm, LL, sum(mm), basis) )
    elseif basis === real 
        if vectorize == false && LL[N] != 0
            error("""For the rSH basis, the CG coefficient is always a vector 
                     except for the case of L=0.""")
        end
        admissible_m = filter( x -> abs(sum(x)) <= LL[N], signed_mmset(mm) )
        C = zeros(ComplexF64, 2*LL[N]+1)
        for __mm in admissible_m
            _mm = SA[__mm...]
            GCG_loc = GCG(ll, _mm, LL, sum(_mm), basis)
            for M_N in signed_m(sum(_mm))
                C[M_N+LL[N]+1] += GCG_loc * 
                                 _Ctran(M_N, sum(_mm), basis)' * 
                                 prod( _Ctran(mm[i], _mm[i], basis) for i in 1:N )
            end
        end

        return LL[N] == 0 ? real(C[1]) : real(C)
    end 
    error("Unknown basis type: $basis")
end

# Function that returns a L set given an `l`. The elements of the set start with 
# l[1] and end with L. 
function SetLl(ll::SVector{N,Int64}, L::Int64) where N
    T = typeof(ll)
    if N==1
        return ll[1] == L ? [T(ll[1])] : Vector{T}[]
    elseif N==2        
        return abs(ll[1]-ll[2]) ≤ L ≤ ll[1] + ll[2] ? [T(ll[1],L)] : Vector{T}[]
    end
    
    set = [ [ll[1];] ]
    for k in 2:N
        set_tmp = set
        set = Vector{Any}[]
        for a in set_tmp
            if k < N
                for b in abs(a[k-1]-ll[k]):a[k-1]+ll[k]
                    push!(set, [a; b])
                end
            elseif k == N
                if (abs.(a[N-1]-ll[N]) <= L)&&(L <= (a[N-1]+ll[N]))
                    push!(set, [a; L])
                end
            end
        end
    end  

    return T.(set)
end

SetLl(ll::SVector{N,Int64}) where N = union([SetLl(ll, L) for L in 0:sum(ll)]...)

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
signed_m(m::T) where T = unique([m,-m])::Vector{T}

# The set of vectors whose i-th element has the same absolute value as m[i] 
# for all i
# signed_mmset(m) = Iterators.product([signed_m(m[i]) for i in 1:length(m)]...
#                                    ) |> collect 

struct LazySignedMMset{N, T} 
    mm::SVector{N, T}
    σ::MVector{N, Bool}
end

lazy_signed_mmset(mm::SVector{N, T}) where {N, T} = 
        LazySignedMMset(mm, zero(MVector{N, Bool}))

function lazy_signed_mmset(mm::Vector{T}) where {T}
    N = length(mm) 
    return lazy_signed_mmset(SVector{N, T}(mm))
end

function Base.iterate(it::LazySignedMMset{N, T}) where {N, T}
    return iterate(it, 0)
end

function Base.iterate(it::LazySignedMMset{N, T}, i::Int) where {N, T}
    i += 1
    if i > 2^N
        return nothing
    end
    σ = it.σ
    digits!(σ, i-1, base=2)
    newmm = SVector(ntuple( 
               j -> ((2*σ[j]-1) * (it.mm[j] != 0) + (it.mm[j] == 0)) * it.mm[j], 
               N )) 
    return newmm, i
end 


function signed_mmset(mm::T, prune=true) where {T}
    N = length(mm); len = 2^N
    MM = Vector{T}(undef, len)
    σ = zeros(Bool, N)
    for i in 1:(2^N)
       digits!(σ, i-1, base=2)
       newmm = [ ((2*σ[j]-1) * (mm[j] != 0) + (mm[j] == 0)) * mm[j]
                 for j = 1:N ] 
       MM[i] = newmm
    end
    if prune
        return unique(MM)
    else
        return MM 
    end
 end

mm_filter_single(mm::Union{Vector{Int64},SVector{N,Int64}}, k::Int64, 
                basis::typeof(complex)) where {N} = (sum(mm) == k)

# for the rSH, the criterion is that whether there exists a combinition 
# of [+/- m_i]_i, such that the sum of the combination equals to k
mm_filter_single(mm::Union{Vector{Int64},SVector{N,Int64}}, k::Int64, 
                basis::typeof(real)) where {N} = 
        any(sum(mm1) == k for mm1 in lazy_signed_mmset(mm))


mm_filter(mm::Union{Vector{Int64},SVector{N,Int64}}, L::Int64,
         basis::typeof(complex)) where {N} = (abs(sum(mm)) <= L)

# for the rSH, the criterion is that whether there exists a combinition 
# of [+/- m_i]_i, such that the sum of the combination equals to k
mm_filter(mm::Union{Vector{Int64},SVector{N,Int64}}, L::Int64,
            basis::typeof(real)) where {N} = 
        any((abs(sum(mm1)) <= L) for mm1 in lazy_signed_mmset(mm))

# Function that generates the set of ordered m's given `n` and `l` with 
# the absolute sum of  m's smaller than or equal to `L`.
#
# NB: This function assumes lexicographical ordering

function mm_generate(L::Int, ll::T, nn::T; 
                     basis = complex) where {T}
    N = length(ll)
    @assert length(ll) == length(nn)

    # No matter PI or not, this fcn always generates all admissible mm's
    # and if PI, they are just filtered in _coupling_coeffs
    # NOTE: this line is the bottleneck in the code, because it generates 
    #       the signed mm set which requires a lot of small allocations.

    # the generator version seems to be type unstable.
    # MM_c = ([ T(I.I) for I in ci if mm_filter(T(I.I), L, basis) ])::Vector{T}
    ci = CartesianIndices(ntuple(t -> -ll[t]:ll[t], N))
    MM_c = T[] 
    for I in ci
        x = T(I.I)
        if mm_filter(x, L, basis)
            push!(MM_c, x)
        end
    end

    if basis === complex
        return MM_c
    elseif basis === real 
        # NOTE: lots of allocations here that could be improved if needed
        MM_abs = unique([ abs.(mm) for mm in MM_c ])
        MM_r = reduce(vcat, signed_mmset(mm, false) for mm in MM_abs)
        return unique(MM_r)
    end
    error("Unknown basis type: $basis")
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
    
    return _coupling_coeffs(_L, _ll, _nn; PI = PI, basis = basis, )
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
                          PI = true, basis = complex) where N


    # NOTE: because of the use of m_generate, the input (nn, ll ) is required
    # to be in lexicographical order.
    nn, ll, inv_perm = lexi_ord(nn, ll)

    Lset = SetLl(ll,L)
    r = length(Lset)
    T = L == 0 ? Float64 : SVector{2L+1,Float64}

    if r == 0; return zeros(T, 0, 0), SVector{N, Int}[]; end

    # there can only be non-trivial coupling coeffs if ∑ᵢ lᵢ + L is even
    if isodd(sum(ll)+L) 
        return zeros(T, 0, 0), SVector{N, Int}[]
    end
     
    if basis === complex 
        if !PI
            MM = mm_generate(L, ll, nn; basis=basis) # all m's
            UMatrix = zeros(T, r, length(MM)) # Matrix containing the coupling coefs D
            for (j,mm) in enumerate(MM)
                for i in 1:r
                    UMatrix[i,j] = GCG(ll,mm,Lset[i];vectorize=(L!=0),basis=basis)
                end
            end 
            return UMatrix, [mm[inv_perm] for mm in MM]
        else
            # permutation blocks - within which the nn and ll are identical
            S = Sn(nn,ll)
            permutable_blocks = [ Vector([S[i]:S[i+1]-1]...) for i in 1:length(S)-1]

            MM = mm_generate(L, ll, nn; basis=basis) # all admissible mm's
            MM_sorted = [ _sort(mm, permutable_blocks) for mm in MM ] # sort the mm's within the permutable blocks
            MM_reduced = unique(MM_sorted) # ordered mm's - representatives of the equivalent classes
            D_MM_reduced = Dict(MM_reduced[i] => i for i in 1:length(MM_reduced))
        
            FMatrix=zeros(T, r, length(MM_reduced)) # Matrix containing f(m,i)

            for (j,mm) in enumerate(MM)
                col = D_MM_reduced[MM_sorted[j]] # avoid looking up the dictionary repeatedly
                for i in 1:r
                    FMatrix[i,col] += GCG(ll,mm,Lset[i];vectorize=(L!=0),basis=basis)
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
    elseif basis === real 
        MM_r = mm_generate(L, ll, nn; basis=basis) # all admissible mm's
        Ure_c, MM_c = _coupling_coeffs(L, ll, nn; PI = false, basis=complex)
        C_r2c = rAA2cAA(SVector{N, Int}.(MM_c),MM_r) 
        # TODO: coupling_coeffs and mm_generate return different 
        #       format of MM's which may need to be fixed
        
        # Do the transformation to the complex coupling 
        # because it has a smaller size compared to the real one
        if L != 0
            CL = SMatrix{2L+1,2L+1}(Matrix(Ctran(L)))
            Ure_c = map(u -> CL * u, Ure_c)
        end
        Ure_r = real(Ure_c * C_r2c)
        
        if !PI
            return Ure_r, [ mm[inv_perm] for mm in MM_r ]
        else
            S = Sn(nn,ll)
            permutable_blocks = [ Vector([S[i]:S[i+1]-1]...) for i in 1:length(S)-1]
            MM_sorted = [ _sort(mm, permutable_blocks) for mm in MM_r ] # sort the mm's within the permutable blocks
            MM_reduced = unique(MM_sorted) # ordered mm's - representatives of the equivalent classes

            # NOTE: this block has a type instability; unclear why.
            D_MM_reduced = Dict{eltype(MM_reduced), Int}(
                    MM_reduced[i] => i for i in 1:length(MM_reduced))
        
            FMatrix=zeros(T, r, length(MM_reduced)) # Matrix containing f(m,i)

            for (j,mm) in enumerate(MM_r)
                col = D_MM_reduced[MM_sorted[j]] # avoid looking up the dictionary repeatedly
                for i in 1:r
                    FMatrix[i,col] += Ure_r[i,j]
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
    error("Unknown basis type: $basis")
end

end
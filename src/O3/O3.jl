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
include("quad_O3_data.jl")
include("quad_O3.jl")



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

function gram(X::AbstractMatrix{SVector{N,T}}) where {N,T}
    G = zeros(T, size(X,1), size(X,1))
    for i = 1:size(X,1)
       for j = i:size(X,1)
          G[i,j] = sum(dot(X[i,t], X[j,t]) for t = 1:size(X,2))
          i == j ? nothing : (G[j,i]=G[i,j]')
       end
    end
    return G
 end

gram(X::AbstractMatrix{<:Number}) = X * X'

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
            # # permutation blocks - within which the nn and ll are identical
            # S = Sn(nn,ll)
            # permutable_blocks = [ Vector([S[i]:S[i+1]-1]...) for i in 1:length(S)-1]

            # MM = mm_generate(L, ll, nn; basis=basis) # all admissible mm's
            # MM_sorted = [ _sort(mm, permutable_blocks) for mm in MM ] # sort the mm's within the permutable blocks
            # MM_reduced = unique(MM_sorted) # ordered mm's - representatives of the equivalent classes
            # D_MM_reduced = Dict(MM_reduced[i] => i for i in 1:length(MM_reduced))
        
            # FMatrix=zeros(T, r, length(MM_reduced)) # Matrix containing f(m,i)

            # for (j,mm) in enumerate(MM)
            #     col = D_MM_reduced[MM_sorted[j]] # avoid looking up the dictionary repeatedly
            #     for i in 1:r
            #         FMatrix[i,col] += GCG(ll,mm,Lset[i];vectorize=(L!=0),basis=basis)
            #     end
            # end 
        
            # # Linear dependence
            # U, S, V = svd(gram(FMatrix))
            # # Somehow rank is not working properly here, might be a relative  
            # # tolerance issue.
            # # original code: rank(Diagonal(S); rtol =  1e-12) 
            # rk = findall(x -> x > 1e-12, S) |> length 
            # # return the RE-PI coupling coeffs
            # return Diagonal(sqrt.(S[1:rk])) * U[:, 1:rk]' * FMatrix, 
            #     [ mm[inv_perm] for mm in MM_reduced ]
            C, MM = coupling_coeffs_new(L, ll, nn)
            return C, [ mm[inv_perm] for mm in MM ]
        end
    elseif basis === real 
        if !PI
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
            return Ure_r, [ mm[inv_perm] for mm in MM_r ]
        else
            S = Sn(nn,ll)
            MM_r = mm_generate(L, ll, nn; basis=basis) # all admissible mm's
            permutable_blocks = [ Vector([S[i]:S[i+1]-1]...) for i in 1:length(S)-1]
            MM_sorted = [ _sort(mm, permutable_blocks) for mm in MM_r ] # sort the mm's within the permutable blocks
            MM_reduced = unique(MM_sorted) # ordered mm's - representatives of the equivalent classes

            Urpe_c, MM_c = _coupling_coeffs(L, ll, nn, PI = PI, basis=complex)
            C_r2c = rAA2cAA_PI(SVector{N, Int}.(MM_c),MM_reduced,MM_r,ll,nn) 
            # TODO: coupling_coeffs and mm_generate return different 
            #       format of MM's which may need to be fixed
            
            # Do the transformation to the complex coupling 
            # because it has a smaller size compared to the real one
            if L != 0
                CL = SMatrix{2L+1,2L+1}(Matrix(Ctran(L)))
                Urpe_c = map(u -> CL * u, Urpe_c)
            end
            Urpe_r = real.(Urpe_c * C_r2c)
            return Urpe_r, [ mm[inv_perm] for mm in MM_reduced ]

            # NOTE: this block has a type instability; unclear why.
            # D_MM_reduced = Dict{eltype(MM_reduced), Int}() 
            # for i in 1:length(MM_reduced)
            #     D_MM_reduced[MM_reduced[i]] = i
            # end
        
            # FMatrix=zeros(T, r, length(MM_reduced)) # Matrix containing f(m,i)

            # for (j,mm) in enumerate(MM_r)
            #     col = D_MM_reduced[MM_sorted[j]] # avoid looking up the dictionary repeatedly
            #     for i in 1:r
            #         FMatrix[i,col] += Ure_r[i,j]
            #     end
            # end 
        
            # # Linear dependence
            # U, S, V = svd(gram(FMatrix))
            # # Somehow rank is not working properly here, might be a relative  
            # # tolerance issue.
            # # original code: rank(Diagonal(S); rtol =  1e-12) 
            # rk = findall(x -> x > 1e-12, S) |> length 
            # # return the RE-PI coupling coeffs
            # return Diagonal(sqrt.(S[1:rk])) * U[:, 1:rk]' * FMatrix, 
            #     [ mm[inv_perm] for mm in MM_reduced ]
        end
    end
    error("Unknown basis type: $basis")
end

## Codes for the new construction

# convert a vector of counts to a feasible mm
function vec2mm(v::Vector{Int})
    L = Int((length(v) - 1)/2)
    N = sum(v) # total number of eggs - correlation order
    mm = Vector{Int}(undef, N)
    idx   = 1
    for (i, count) in enumerate(v)
        val = i - L - 1
        for _ in 1:count
            mm[idx] = val
            idx += 1
        end
    end
    return mm
end

# make a feasible mm to a counting of elements in the vector
function mm2vec(mm::AbstractVector{<:Integer}, L::Integer)
    v = zeros(Int, 2L+1)
    @inbounds for m in mm
        v[m+L+1] += 1
    end
    return v
end

# the value of derivative wrt beta at the origin for certain l,m,μ
db(l::Int,m::Int,μ::Int) = m - μ == 1 ? -1/2*((l-μ)*(l+m))^(1/2) : m - μ == -1 ? 1/2*((l+μ)*(l-m))^(1/2) : 0

function right_shift_neighbors(A::AbstractVector{<:Integer})
    m = length(A)
    l = Int((m-1)/2)

    # number of neighbours - only those position n so that A[n] ≥ 1 can be shifted to the right
    k = count(@view(A[1:end-1])) do x
                 x ≥ 1
             end

    neighbors = Vector{Vector{eltype(A)}}(undef, k)   # pre-allocate
    val = Vector{Float64}(undef, k)

    j = 1
    @inbounds for n in 1:m-1
        if A[n] ≥ 1
            B = copy(A) # make a copy of A
            B[n] -= 1
            B[n+1] += 1
            neighbors[j] = B
            val[j] = B[n+1] * db(l, n-l-1, n-l)
            j += 1
        end
    end
    return neighbors, val
end

function left_shift_neighbors(A::AbstractVector{<:Integer})
    m = length(A)
    l = Int((m-1)/2)

    # number of neighbours
    k = count(@view(A[2:end])) do x
                 x ≥ 1
             end

    neighbors = Vector{Vector{eltype(A)}}(undef, k)   # pre-allocate neighboring vectors
    val = Vector{Float64}(undef, k) # pre-allocate matrix elements 

    j = 1
    @inbounds for n in 2:m
        if A[n] ≥ 1
            B = copy(A) # make a copy of A
            B[n] -= 1
            B[n-1] += 1
            neighbors[j] = B
            val[j] = B[n-1] * db(l, n-l-1, n-l-2)
            j += 1
        end
    end
    return neighbors, val
end

function efficient_cartesian_concat(mmset_sep::Vector{Vector{Vector{Int}}})
    len = length(mmset_sep) # number of blocks
    sizes = map(length, mmset_sep)
    total = prod(sizes) # predict the final length
    
    # Precompute block sizes for indexing
    block_sizes = [prod(sizes[i+1:end]) for i in 1:len]
    
    # Precompute result vector lengths
    block_lengths = map(v -> length(v[1]), mmset_sep)  # assuming uniform length within each block
    total_length = sum(block_lengths)

    mmset = Vector{Vector{Int}}(undef, total)

    for i in 0:total-1
        temp = Vector{Int}(undef, total_length)
        pos = 1
        idx = i
        for d in 1:len
            q, r = divrem(idx, block_sizes[d])
            vec = mmset_sep[d][q+1]
            for v in vec
                temp[pos] = v
                pos += 1
            end
            idx = r
        end
        mmset[i+1] = temp
    end

    return mmset
end

# TODO: I guess I should swap mm and μμ to make the notation more consistent as before
# In addition, in the function mat, the matrix is defined row-wise (Fig (1) in the manuscript). 
function mat(K::Int,ll::AbstractVector{Int},nn::AbstractVector{Int})
    idx = Sn(nn,ll) # separable blocks
    lset = ll[idx[1:end-1]] # l's of the blocks
    nset = [ idx[i] - idx[i-1] for i in 2:length(idx) ] # lengths of the blocks

    len = length(lset) # number of blocks
    @assert length(lset) == length(nset)
    
    # @time mmset_sep = [ vec2mm.(sep(lset[i],nset[i])) for i in 1:len ] # separated mm's for each l and N
    mmset_sep = [all_mm(lset[i], nset[i]) for i in 1:length(lset)]  # separated mm's for each l and N - equivalent to the above but is faster
    mmset = efficient_cartesian_concat(mmset_sep) # cartesian product of mmset_sep

    μμset = mmset[findall(x -> abs(sum(x)) <= K, mmset)]
    # mmset = K != 0 ? mmset[findall(x -> abs(sum(x)) <= K + 1, mmset)] : mmset[findall(x -> abs(sum(x)) == K + 1, mmset)]
    mmset = K != 0 ? mmset[findall(x -> ((-K <= sum(x) <= K - 1)||(sum(x) == K + 1)), mmset)] : mmset[findall(x -> sum(x) == K + 1, mmset)]

    # μμset = sort(μμset, by = x -> (sum(x), x)) # Sort μμset by the sum of elements in μμ, and then lexicographically
    # mmset = sort(mmset, by = x -> (sum(x), x)) # Sort mmset by the sum of elements in mm, and then lexicographically
    # μμset = blockwise_sort(μμset, nset) # Sort μμset by the sum of elements in μμ, and then the sum of each block, and finally lexicographically
    # mmset = blockwise_sort(mmset, nset) # Sort mmset by the sum of elements in mm, and then the sum of each block, and finally lexicographically
    sort!(μμset, by = sum) # Sort μμset by the sum of elements in μμ, and then lexicographically
    sort!(mmset, by = sum) # Sort mmset by the sum of elements in mm, and then lexicographically

    dict_μμ = Dict{Vector{Int}, Int}(μμset[i] => i for i in 1:length(μμset))

    # r = 0 # Number of rows in the matrix
    # for mm in mmset
    #     r += 1
    #     # push!(mmset_aug, mm)
    #     if sum(mm) >= -K + 1 && sum(mm) <= K - 1 # mm within this range produces two rows
    #         r += 1
    #     end
    # end
    # r = Int(r/2)

    a = 0 # Calculating the number of possible rows and compare in the end

    # triplets to generate the sparse matrix M
    mm_idx = Int[]
    μμ_idx = Int[]
    vals = Float64[]

    # NOTE: r is the exact column that we know; for each mm in mmset, 
    # based on left right shifts, we can have at most 2K μμ's, and including 
    # mm itself, we can have at most 2K+1 non-zero entries in each column
    # sizehint!(mm_idx, (2K+1)*r) 
    # sizehint!(μμ_idx, (2K+1)*r)
    # sizehint!(vals, (2K+1)*r)
    # But this estimation is loose.

    μμ = similar(mmset[1]) # allocate μμ once
    for mm in mmset
        S = sum(mm)
        if -K <= S <= K-1
            a += 1
            I = S + 1
            block_end = 0
            for (len_loop, l) in enumerate(lset)
                block_start = block_end + 1
                block_end  += nset[len_loop] # indices for the current ll block
                @views mm_loc = mm[block_start:block_end]
                vec_mm_loc = mm2vec(mm_loc,l)
                vec_μμset_loc, val = right_shift_neighbors(vec_mm_loc)

                copy!(μμ,mm) # create a copy of mm to modify it
                j = dict_μμ[μμ]
                push!(mm_idx, a)
                push!(μμ_idx, j)
                push!(vals, -db(K, I-1, I))

                for (t,vec_μμ_loc) in enumerate(vec_μμset_loc)
                    μμ[block_start:block_end] = vec2mm(vec_μμ_loc) # modify the μμ copy
                    j = dict_μμ[μμ]

                    push!(mm_idx, a)
                    push!(μμ_idx, j)
                    push!(vals, val[t])
                end
            end
        elseif S == K + 1
            # This B^+ matrix is independent of the above system (bottom right single block
            # in Fig (1) of the manuscript), so the sign of this is actually not important. 
            # And since this is the only part that call db in the first case, 
            # the sign in db doesn't matter for the correctness of the code, 
            # but we should keep it to be the correct one.
            a += 1
            I = K
            block_end = 0
            for (len_loop, l) in enumerate(lset)
                block_start = block_end + 1
                block_end  += nset[len_loop] # indices for the current ll block
                @views mm_loc = mm[block_start:block_end]
                vec_mm_loc = mm2vec(mm_loc,l)

                vec_μμset_loc, val = left_shift_neighbors(vec_mm_loc)

                copy!(μμ,mm)
                for (t,vec_μμ_loc) in enumerate(vec_μμset_loc)
                    μμ[block_start:block_end] = vec2mm(vec_μμ_loc) # modify the μμ copy
                    j = dict_μμ[μμ]

                    push!(mm_idx, a)
                    push!(μμ_idx, j)
                    push!(vals, val[t])
                end
            end
        else
            continue
        end               
    end

    @assert a == length(mmset) # Check that the number of rows is equal to the number of filtered mm's
    triplets = unique(zip(mm_idx, μμ_idx, vals))
    mm_idx = [i for (i, j, v) in triplets]
    μμ_idx = [j for (i, j, v) in triplets]
    vals = [v for (i, j, v) in triplets]

    mat = sparse(μμ_idx, mm_idx, vals, length(μμset), a) # Create a sparse matrix from the indices and values
    # mat = sparse(mm_idx, μμ_idx, vals, a, length(μμset)) # Create a sparse matrix from the indices and values

    return mat, μμset, mmset # mat[1:a,:] # [mat; mat_minus; mat_plus]
end

function nullspace_upper_sparse(U::AbstractMatrix{T}) where T<:Number
    m, n = size(U)
    @assert m ≤ n # U must have at least as many columns as rows
    r = n - m # rank of the null
    r == 0 && return zeros(T, n, 0) # full column rank → trivial nullspace

    # [ U_square; U_reduced ] dot [X I] = 0 <=> U_square X + U_reduced = 0
    RHS = -Matrix(U[1:m, m+1:n])
    X = UpperTriangular(U[1:m, 1:m]) \ RHS   # dense or sparse, Julia picks the right solver

    # [X; I] is the basis of the null space
    N = zeros(T, n, r)
    N[1:m, :] .= X
    @inbounds for j in 1:r
        N[m + j, j] = one(T)
    end
    return N ./ norm(N)
end

function solver_inner(M::AbstractMatrix{T}, mmset::Vector{Vector{Int}}, μμset::Vector{Vector{Int}}) where T<:Number
    M = sparse(M')
    C = zeros(Float64, size(M, 2), size(M, 2) - size(M, 1))

    row_sum = sum.(mmset)
    column_sum = sum.(μμset)

    row_range = findall(i -> i == 1 || row_sum[i] ≠ row_sum[i-1], 1:length(mmset))
    column_range = findall(i -> i == 1 || column_sum[i] ≠ column_sum[i-1], 1:length(μμset))
    push!(row_range, length(mmset) + 1)
    push!(column_range, length(μμset) + 1)

    # The lowest block
    row_block = row_range[end-1]:row_range[end]-1
    prev_col_block = column_range[end-1]:column_range[end]-1

    # resolution to its kernel
    if length(row_range) == length(column_range)
        B = M[row_block, prev_col_block]
        F = lu(B')
        invp = invperm(F.p)
        sparse_ns = nullspace_upper_sparse(sparse(F.L'))
        C[prev_col_block, :] .= Matrix((F.Rs .* sparse_ns[invp,:])')'


        # Back-substitution
        for t in length(row_range)-2:-1:1
            row_block = row_range[t]:row_range[t+1]-1
            curr_col_block = column_range[t]:column_range[t+1]-1
            
            B = M[row_block, prev_col_block] # B matrix for the current block row and the previous column block
            C_prev = @view C[prev_col_block, :] # The block we computed in the previous iteration
            C_curr  = @view C[curr_col_block, :]  # The block we are computing right now
            
            # The scalar from A
            a = M[row_block[1], curr_col_block[1]] 
            
            # C_curr = B * C_prev / a + 0
            mul!(C_curr, B, C_prev, -1.0/a, 0.0)
            
            # Shift column
            prev_col_block = curr_col_block
        end
    else # the case where \sum ll = L
        for (i, col_idx) in enumerate(prev_col_block)
            C[col_idx, i] = 1.0
        end


        # Back-substitution
        for t in length(row_range)-1:-1:1
            row_block = row_range[t]:row_range[t+1]-1
            curr_col_block = column_range[t]:column_range[t+1]-1
            
            B = M[row_block, prev_col_block] # B matrix for the current block row and the previous column block
            C_prev = @view C[prev_col_block, :] # The block we computed in the previous iteration
            C_curr  = @view C[curr_col_block, :]  # The block we are computing right now
            
            # The scalar from A
            a = M[row_block[1], curr_col_block[1]] 
            
            # C_curr = B * C_prev / a + 0
            mul!(C_curr, B, C_prev, -1.0/a, 0.0)
            
            # Shift column
            prev_col_block = curr_col_block
        end
    end
    return C
end

function coupling_coeffs_new(K::Int, ll::AbstractVector{<:Int}, nn::AbstractVector{<:Int})
    # TODO: notation inconsistency: K and L both represent the order of equivariance
    # TODO: reconsider if ll and nn here should be made to be SVector{N, Int}
    T = K == 0 ? Float64 : SVector{2K+1,Float64}
    N = length(ll)
    @assert length(ll) == length(nn)

    if K > sum(ll) # if the matrix is square, we can use the nullspace function
        # return null_return(Val(K),μμset) # Matrix{SVector{2K+1, Float64}}(undef, 0, length(μμset)), μμset
        # return Matrix{SVector{2K+1, Float64}}(undef, 0, length(μμset)), μμset
        return zeros(T, 0, 0), SVector{N, Int}[]
    end

    if all(iszero, ll) && K == 0 # Only in this trivial case we don't have the following matrix
        return [1;;], [SVector{N, Int}(zeros(N)), ]
    end
    
    M, μμset, mmset = mat(K, ll, nn) # The matrix that we will use to compute the RPI basis

    if size(M,1) == size(M,2) # if the matrix is square, we return 0 - but this can only happen when K > sum(ll), which is already handled in the beginning of the function, so this is just for safety
        # return null_return(Val(K),μμset) # Matrix{SVector{2K+1, Float64}}(undef, 0, length(μμset)), μμset
        # return Matrix{SVector{2K+1, Float64}}(undef, 0, length(μμset)), μμset
        return zeros(T, 0, 0), SVector{N, Int}[]
    end

    # method I: find the null space of the matrix M using nullspace
    # C = Matrix(transpose(nullspace(M))) # Use the full matrix to compute the null space

    # method II: use QR decomposition to find the null space
    # F = qr(M)
    # C = Matrix(F.Q[invperm(F.prow), end-(size(M,1)-size(M,2))+1:end]') # we know the rank already :)
    
    # method III: use lu factorization to find the null space - cf old code
    # F = lu(M)
    # invp = invperm(F.p)
    # sparse_ns = nullspace_upper_sparse(sparse(F.L'))
    # C = Matrix((F.Rs .* sparse_ns[invp,:])')

    # method IV: the new solver fully corresponds to the paper - back-substitution!
    C = solver_inner(M, mmset, μμset)
    
    if K == 0
        return C', μμset
    end

    return embed_in_onehot(C', μμset, K), μμset # embed the null space in one-hot vectors
end

function embed_in_onehot(C::AbstractMatrix,mmset::AbstractVector, K::Integer)
    m, n = size(C)
    T = eltype(C)
    Vec = SVector{2K+1, T}

    # one-hot vectors for every column
    colvec = Vector{Vec}(undef, n)
    z = zeros(Vec)
    for j in 1:n
        idx = sum(mmset[j]) + K + 1
        # colvec[j] = Vec(ntuple(k -> (k == idx ? one(T) : zero(T)), 2K + 1))
        colvec[j] = setindex(z, one(T), idx)
    end

    # broadcast the multiplication
    return broadcast(*, C, reshape(colvec, 1, n))::Matrix{Vec}
end

all_mm(l::Int, N::Int) = collect(with_replacement_combinations(-l:l, N))

end
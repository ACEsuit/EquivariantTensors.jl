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

# A new structure for efficiently constructing PermutableBlocks for ordered (nn,ll)
# E.g., for nn = [1,2,1,1,2], ll = [1,1,2,2,2], the permutable blocks
# are 1:1, 2:2, 3:4, 5:5.
# To see that, run: for pb in PermutableBlocks(nn,ll); @show pb; end
struct PermutableBlocks{N, T1, T2}
    nn::SVector{N, T1}
    ll::SVector{N, T2}
end

# iterate over PermutableBlocks
function Base.iterate(iter::PermutableBlocks{N}, state=1) where N
    state > N && return nothing
    
    start_idx = state
    
    # find block ends
    @inbounds for i in (start_idx + 1):N
        if iter.ll[i] != iter.ll[start_idx] || iter.nn[i] != iter.nn[start_idx]
            # return the UnitRange and pass i as the state for the next iteration
            return (start_idx:(i - 1), i)
        end
    end
    
    # return when reaching the very end
    return (start_idx:N, N+1)
end
get_permutable_blocks(nn::SVector{N,T}, ll::SVector{N,T}) where {N,T} = PermutableBlocks(nn, ll)

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

function signed_mmset(MM_abs::AbstractArray{SVector{N,Int}}, prune=true) where N
    MM = SVector{N,Int}[]
    for mm in MM_abs

        if prune
            idx = findall(!iszero, mm) # index for nonzero entries
        else
            idx = 1:N # when !prune, all position is allowed to flip
        end

        for k in 1:2^(length(idx)) # all possibilities - this is the prune case as before
            μμ = copy(mm)
            # flip (or not) the allowed positions
            for (i, j) in pairs(idx)
                if div(k-1, 2^(i-1)) % 2 == 1 # determine whether to flip
                    μμ = setindex(μμ, -μμ[j], j)
                end
            end
            push!(MM, μμ)
        end
    end
    return MM
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

# Generate mm depending on PI, for !PI this fcn always generates all admissible mm's
# and if PI, it generates ordered mm wrt input ll and nn for basis = complex
# and for basis = real, it generates all mm s.t. abs.(mm) in abs.(MM_c)
# ordering will be done in rAA2cAA_PI to avoid duplicate sorting
# NOTE: this line seems not to be the bottleneck in the code anymore

function mm_generate(L::Int, ll::SVector{N,Int}, nn::SVector{N,Int}; 
                     basis = complex, PI = false) where {N}
    # the generator version seems to be type unstable.
    # MM_c = ([ T(I.I) for I in ci if mm_filter(T(I.I), L, basis) ])::Vector{T}
    if !PI
        # LIWEI: I think there might be a faster way to do this, but the !PI case in not my focus
        ci = CartesianIndices(ntuple(t -> -ll[t]:ll[t], N))
        MM_c = SVector{N,Int}[] 
        for I in ci
            x = SVector{N,Int}(I.I)
            if mm_filter(x, L, basis)
                push!(MM_c, x)
            end
        end
    elseif PI
        permutable_blocks = get_permutable_blocks(nn, ll)
        lset = Int[] # l's of the blocks
        nset = Int[] # lengths of the blocks

        for block in permutable_blocks
            push!(lset, ll[first(block)])
            push!(nset, length(block))
        end

        len = length(lset) # number of blocks
        @assert length(lset) == length(nset)

        MM_c = all_mm_blocks(lset,nset,Val(N))
    end

    if basis === complex
        return MM_c
    elseif basis === real 
        # NOTE: lots of allocations here that could be improved if needed
        # LIWEI: This seems to have been improved
        MM_abs = unique([ abs.(mm) for mm in MM_c ])
        return signed_mmset(MM_abs)
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

function lexi_ord(nn::SVector{N, Int}, ll::SVector{N, Int}) where N
   bb = [ (ll[i], nn[i]) for i = 1:N ]
   p = sortperm(bb)
   bb_sorted = bb[p]
   return SVector{N, Int}(ntuple(i -> bb_sorted[i][2], N)), 
          SVector{N, Int}(ntuple(i -> bb_sorted[i][1], N)), 
          SVector{N, Int}(invperm(p))
end

"""
    O3.coupling_coeffs(L, ll, nn; PI, basis, refl_sym)
    O3.coupling_coeffs(L, ll; PI, basis, refl_sym)

Compute coupling coefficients for the spherical harmonics basis, where 
- `L` must be an `Integer`;
- `ll, nn` must be vectors or tuples of `Integer` of the same length.
- `PI`: whether or not the coupled basis is permutation-invariant (or the 
corresponding tensor symmetric); default is `true` when `nn` is provided 
and `false` when `nn` is not provided.
- `basis`: which basis is being coupled, default is `complex`, alternative
choice is `real`, which is compatible with the `SpheriCart.jl` convention.  
- `refl_sym`: behaviour of the basis under reflection, options are Symbols
`:sym` and `:asym`, which indicate reflection symmetry and anti-symmetry, resp.
default is `nothing`, which will later be assigned as `sym` for even `L` and 
`asym` for odd `L`.  
"""
function coupling_coeffs(L::Integer, ll, nn = nothing; 
                         PI = !(isnothing(nn)), 
                         basis = complex,
                         refl_sym::Union{Symbol,Nothing} = nothing)

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
    
    return _coupling_coeffs(_L, _ll, _nn; PI = PI, basis = basis, refl_sym = refl_sym)
end

function _sort(x::SVector{N,T}, permutable_blocks::PermutableBlocks) where {N,T}
    # Sorts the vector x according to the indices in permutable_blocks
    # This is used to sort the equivalent classes of m's
    x = MVector(x)
    
    for block in permutable_blocks
        @views sort!(x[block])
    end
    
    return SVector{N,T}(x)
end


# Function that generates the coupling coefficient of the RE basis (PI = false) 
# or RPE basis (PI = true) given `nn` and `ll`. 
function _coupling_coeffs(L::Int, ll::SVector{N, Int}, nn::SVector{N, Int}; 
                          PI = true, basis = complex, refl_sym::Union{Symbol,Nothing} = nothing) where N
    refl_sym === nothing && (refl_sym = iseven(L) ? :sym : :asym)
    if refl_sym == :sym
        ll_filter = iseven
    elseif refl_sym == :asym
        ll_filter = isodd
    else
        error("Unknown reflection symmetry type: $refl_sym")
    end

    # NOTE: because of the use of m_generate, the input (nn, ll ) is required
    # to be in lexicographical order.
    nn, ll, inv_perm = lexi_ord(nn, ll)
    T = L == 0 ? Float64 : SVector{2L+1,Float64}

    # there can only be non-trivial coupling coeffs if ∑ᵢ lᵢ + L is even
    if !ll_filter(sum(ll)) 
        return zeros(T, 0, 0), SVector{N, Int}[]
    end
     
    if basis === complex 
        if !PI
            # TODO: The function SetLl is not type stable.
            # If we implement the !PI case also with the new method,
            # it be removed entirely.
            Lset = SetLl(ll,L)
            r = length(Lset)
            if r == 0; return zeros(T, 0, 0), SVector{N, Int}[]; end

            MM = mm_generate(L, ll, nn; basis=basis) # all m's
            UMatrix = zeros(T, r, length(MM)) # Matrix containing the coupling coefs D
            for (j,mm) in enumerate(MM)
                for i in 1:r
                    UMatrix[i,j] = GCG(ll,mm,Lset[i];vectorize=(L!=0),basis=basis)
                end
            end 
            return UMatrix, [mm[inv_perm] for mm in MM]
        else
            # Old method is commented out
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
            Ure_c, MM_c = _coupling_coeffs(L, ll, nn; PI = false, basis=complex, refl_sym = refl_sym)
            C_r2c = rAA2cAA(SVector{N, Int}.(MM_c),MM_r) 
            # TODO: coupling_coeffs and mm_generate return different 
            #       format of MM's which may need to be fixed
            
            # Do the transformation to the complex coupling 
            # because it has a smaller size compared to the real one
            if L != 0
                CL = SMatrix{2L+1,2L+1}(Matrix(Ctran(L)))
                Ure_c = map(u -> CL * u, Ure_c)
            end
            if norm(real(Ure_c * C_r2c) - Ure_c * C_r2c)≤1e-12
                Ure_r = real(Ure_c * C_r2c) 
            else
                Ure_r = Ure_c * C_r2c 
                @warn("Non-real couplings for L = $L, ll = $ll, nn = $nn, refl_sym = $refl_sym, nob = $(size(Ure_c,1))")
            end
            return Ure_r, [ mm[inv_perm] for mm in MM_r ]
        else
            # S = Sn(nn,ll)
            MM_r = mm_generate(L, ll, nn; basis=basis, PI = true) # all admissible mm's wrt ordered cSH mm's
            Urpe_c, MM_c = _coupling_coeffs(L, ll, nn, PI = PI, basis=complex, refl_sym = refl_sym) # cSH-based couplings
            C_r2c, MM_reduced = rAA2cAA_PI(SVector{N, Int}.(MM_c),SVector{N, Int}.(MM_r),ll,nn) # r2c map and the ordered mm set
            # TODO: coupling_coeffs and mm_generate return different 
            #       format of MM's which may need to be fixed
            
            # Do the transformation to the complex coupling 
            # because it has a smaller size compared to the real one
            
            # Combine the two operations 
            CL = SMatrix{2L+1,2L+1}(Matrix(Ctran(L)))
            Urpe_r = assemble_U(Urpe_c, CL, C_r2c)
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

function assemble_U(U::AbstractMatrix{SVector{L, Float64}}, 
                          CL::SMatrix{L, L, ComplexF64}, 
                          C::SparseMatrixCSC{ComplexF64, Int}) where L
    a, b = size(U)
    _, c = size(C)
    
    # Preallocate the final matrix (we know its real)
    R = zeros(SVector{L, Float64}, a, c)
    
    # Allocate a small workspace for a single column
    W = zeros(SVector{L, ComplexF64}, a)
    
    # Extract real and imaginary parts of CL
    # to skip calculating the imaginary part of (CL * W), which we know is 0
    CL_R = real.(CL)
    CL_I = imag.(CL)
    
    rows = rowvals(C)
    vals = nonzeros(C)  
    
    for j in 1:c
        fill!(W, zero(SVector{L, ComplexF64}))
        
        # W = U * C[:, j]
        for p in nzrange(C, j)
            k = rows[p]
            v = vals[p]
            
            for i in 1:a
                W[i] += U[i, k] * v
            end
        end
        
        # Multiply CL and extract real part: R[:, j] = Re(CL * W)
        # We use Re(CL * W) = CL_R * Re(W) - CL_I * Im(W) to halve the operations
        for i in 1:a
            w = W[i]
            R[i, j] = CL_R * real(w) - CL_I * imag(w)
        end
    end
    
    return R
end

# Case L = 0
function assemble_U(U::AbstractMatrix{Float64}, 
                          CL::AbstractMatrix,
                          C::SparseMatrixCSC{ComplexF64, Int})
    a, b = size(U)
    _, c = size(C)
    
    R = zeros(Float64, a, c)
    
    rows = rowvals(C)
    vals = nonzeros(C)
    
    for j in 1:c
        for p in nzrange(C, j)
            k = rows[p]
            vr = real(vals[p]) 
            for i in 1:a
                R[i, j] += U[i, k] * vr
            end
        end
    end
    
    return R
end


## Codes for the new construction

# the value of derivative wrt beta at the origin for certain l,m,μ
db(l::Int,m::Int,μ::Int) = m - μ == 1 ? -0.5 * sqrt((l-μ) * (l+m)) : m - μ == -1 ? 0.5 * sqrt((l+μ) * (l-m)) : 0.0

# TODO: I guess I should swap mm and μμ to make the notation more consistent as before
# In addition, in the function mat, the matrix is defined row-wise (Fig (1) in the manuscript). 
function mat(K::Int,ll::SVector{N,Int},nn::SVector{N,Int}) where N
    permutable_blocks = get_permutable_blocks(nn, ll)
    lset = Int[] # l's of the blocks
    nset = Int[] # lengths of the blocks

    for block in permutable_blocks
        push!(lset, ll[first(block)])
        push!(nset, length(block))
    end

    len = length(lset) # number of blocks
    @assert length(lset) == length(nset)
    
    # # @time mmset_sep = [ vec2mm.(sep(lset[i],nset[i])) for i in 1:len ] # separated mm's for each l and N
    # mmset_sep = [all_mm(lset[i], nset[i]) for i in 1:length(lset)]  # separated mm's for each l and N - equivalent to the above but is faster
    # mmset = efficient_cartesian_concat(mmset_sep) # cartesian product of mmset_sep
    mmset = mm_generate(K, ll, nn; basis = complex, PI = true)

    μμset = mmset[findall(x -> abs(sum(x)) <= K, mmset)]
    # mmset = K != 0 ? mmset[findall(x -> abs(sum(x)) <= K + 1, mmset)] : mmset[findall(x -> abs(sum(x)) == K + 1, mmset)]
    mmset = K != 0 ? mmset[findall(x -> ((-K <= sum(x) <= K - 1)||(sum(x) == K + 1)), mmset)] : mmset[findall(x -> sum(x) == K + 1, mmset)]

    # μμset = sort(μμset, by = x -> (sum(x), x)) # Sort μμset by the sum of elements in μμ, and then lexicographically
    # mmset = sort(mmset, by = x -> (sum(x), x)) # Sort mmset by the sum of elements in mm, and then lexicographically
    # μμset = blockwise_sort(μμset, nset) # Sort μμset by the sum of elements in μμ, and then the sum of each block, and finally lexicographically
    # mmset = blockwise_sort(mmset, nset) # Sort mmset by the sum of elements in mm, and then the sum of each block, and finally lexicographically
    sort!(μμset, by = sum) # Sort μμset by the sum of elements in μμ, and then lexicographically
    sort!(mmset, by = sum) # Sort mmset by the sum of elements in mm, and then lexicographically

    dict_μμ = Dict{SVector{N,Int}, Int}(μμset[i] => i for i in 1:length(μμset))

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

    for mm in mmset
        S = sum(mm)
        if -K <= S <= K-1
            a += 1
            I = S + 1

            j = dict_μμ[mm]
            push!(mm_idx, a)
            push!(μμ_idx, j)
            push!(vals, -db(K, I-1, I)) # Element in A^-
            
            block_end = 0            
            for (l, n) in zip(lset, nset)
                block_start = block_end + 1
                block_end  += n # indices for the current ll block
    
                # Iterate backwards to easily find the LAST occurrence of each unique state
                for i in block_end:-1:block_start
                    m = mm[i]
                    
                    if m < l && (i == block_end || m < mm[i+1]) # where we can add by 1
                        μμ = setindex(mm,m+1,i)
                        j = dict_μμ[μμ]

                        λ = count(==(μμ[i]), view(μμ, block_start:block_end))
                        val = λ * db(l, μμ[i]-1, μμ[i])

                        push!(mm_idx, a)
                        push!(μμ_idx, j)
                        push!(vals, val) # element in B^-
                    end
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
            for (l, n) in zip(lset, nset)
                block_start = block_end + 1
                block_end  += n # indices for the current ll block

                for i in block_start:block_end
                    m = mm[i]
                    
                    if m > -l && (i == block_start || m > mm[i-1]) # where we can subtract by 1
                        μμ = setindex(mm,mm[i]-1,i)
                        j = dict_μμ[μμ]

                        λ = count(==(μμ[i]), view(μμ, block_start:block_end))
                        val = λ * db(l, μμ[i]+1, μμ[i])

                        push!(mm_idx, a)
                        push!(μμ_idx, j)
                        push!(vals, val)  # element in B^+
                    end
                end
            end
        else
            continue
        end               
    end

    @assert a == length(mmset) # Check that the number of rows is equal to the number of filtered mm's
    M = sparse(μμ_idx, mm_idx, vals, length(μμset), a) # Create a sparse matrix from the indices and values

    return M, μμset, mmset
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

function solver_inner(M::AbstractMatrix{T}, mmset::Vector{SVector{N,Int}}, μμset::Vector{SVector{N,Int}}) where {N,T<:Number}
    M = sparse(M')
    C = zeros(Float64, size(M, 2) - size(M, 1), size(M, 2))

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
    # Solving for the last block, cf Fig 1 (b,c)
    if length(row_range) == length(column_range)
        B = M[row_block, prev_col_block]
        F = lu(B')
        invp = invperm(F.p)
        sparse_ns = nullspace_upper_sparse(sparse(F.L'))
        C[:, prev_col_block] .= (Diagonal(F.Rs) * sparse_ns[invp, :])'
        t_start = length(row_range)-2
    else
        for (i, col_idx) in enumerate(prev_col_block)
            C[i, col_idx] = 1.0
        end
        t_start = length(row_range)-1
    end

    # Back-substitution
    for t in t_start:-1:1
        row_block = row_range[t]:row_range[t+1]-1
        curr_col_block = column_range[t]:column_range[t+1]-1
        
        a = -1/M[row_block[1], curr_col_block[1]] 
        
        # Extract the internal sparse arrays for maximum speed
        colptr = M.colptr
        rowval = M.rowval
        nzval  = M.nzval
        
        # Perform C_curr = B * C_prev directly without allocating B!
        # Because we are using C_T, curr_col_block and prev_col_block are COLUMNS
        for (j_idx, j_global) in enumerate(prev_col_block)
            
            # Loop only over the non-zeros in column j_global of M_sparse
            for p in colptr[j_global]:(colptr[j_global+1]-1)
                i_global = rowval[p]
                
                # If this non-zero falls inside our row_block, do the math
                if i_global >= row_block[1] && i_global <= row_block[end]
                    i_idx = i_global - row_block[1] + 1
                    
                    val = nzval[p] * a
                    
                    # Multiply into our contiguous C_T array
                    # This replaces mul! and is 100% allocation-free
                    for k in 1:size(M, 2) - size(M, 1)
                        C[k, curr_col_block[i_idx]] += val * C[k, prev_col_block[j_idx]]
                    end
                end
            end
        end
        
        prev_col_block = curr_col_block
    end
    
    # Transpose back at the very end to return your expected dimensions!
    return C
end

# Core function that generates the L-equivariant CCs for ordered (nn,ll)
function coupling_coeffs_new(K::Int, ll::SVector{N,Int}, nn::SVector{N,Int}) where N
    # TODO: notation inconsistency: K and L both represent the order of equivariance
    # TODO: reconsider if ll and nn here should be made to be SVector{N, Int}
    T = K == 0 ? Float64 : SVector{2K+1,Float64}
    @assert length(ll) == length(nn)

    if K > sum(ll) # if the matrix is square, we can use the nullspace function
        # return null_return(Val(K),μμset) # Matrix{SVector{2K+1, Float64}}(undef, 0, length(μμset)), μμset
        # return Matrix{SVector{2K+1, Float64}}(undef, 0, length(μμset)), μμset
        return zeros(T, 0, 0), SVector{N, Int}[]
    end

    if all(iszero, ll) && K == 0 # Only in this trivial case we don't have the following matrix
        return [1.0;;], [SVector{N, Int}(zeros(N)), ]
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
        return C, μμset
    end

    return embed_in_onehot(C, μμset, Val(K)), μμset # embed the null space in one-hot vectors
end

function embed_in_onehot(C::AbstractMatrix{T}, mmset::AbstractVector, ::Val{K}) where {T, K}
    m, n = size(C)
    Vec = SVector{2K+1, T} 
    
    # Allocate the final matrix of vectors - the memory cost here seems to be unavoidable
    C_vec = Matrix{Vec}(undef, m, n)
    z = zeros(Vec)
    
    for j in 1:n
        idx = sum(mmset[j]) + K + 1
        for i in 1:m
            C_vec[i, j] = setindex(z, C[i,j], idx)
        end
    end
    
    return C_vec
end

@inline function _fill_mm!(out, current, lset, nset, block, local_depth, global_depth, min_val, idx)
    # 1. Base Case: The vector is completely filled
    if global_depth > length(current)
        @inbounds out[idx] = SVector(current)
        return idx + 1
    end

    # 2. Block Transition: Current block is full, jump to the next block
    if local_depth > @inbounds nset[block]
        next_l = @inbounds lset[block + 1]
        return _fill_mm!(out, current, lset, nset, block + 1, 1, global_depth, -next_l, idx)
    end

    # 3. Recursive Step: Iterate valid numbers and dive deeper
    l = @inbounds lset[block]
    for v in min_val:l
        @inbounds current[global_depth] = v
        idx = _fill_mm!(out, current, lset, nset, block, local_depth + 1, global_depth + 1, v, idx)
    end

    return idx
end
"""
all_mm_blocks(lset::AbstractVector{Int}, nset::AbstractVector{Int}, ::Val{N}) where N

Generate the set of equivalent classes given ll with minimal partition
[ repeat(lset[i], nset[i]) for i = 1:Nblock ]

N = sum(nset) should be given to make the size of the output explicit.
"""
# A version that gets rid of the Cartesian product, avoiding type
# instablity in _coupling_coeffs.
function all_mm_blocks(lset::AbstractVector{Int},nset::AbstractVector{Int},::Val{N}) where N
    @assert length(lset) == length(nset)
    @assert sum(nset) == N

    # pre-allocate since we know the size
    total = prod(binomial(2*l+n, n) for (l,n) in zip(lset,nset))
    out = Vector{SVector{N, Int}}(undef, total)
    current = MVector{N, Int}(undef)

    _fill_mm!(out, current, lset, nset, 1, 1, 1, -lset[1], 1)
    # _fill_blocks!(out, current, lset, nset, 1, 0, 1)
    return out
end

end
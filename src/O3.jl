module O3
 
using PartialWaveFunctions
using Combinatorics
using LinearAlgebra
using StaticArrays
using SparseArrays

export coupling_coeffs

struct B_cSH end 
struct B_SpheriCart end 
struct B_CondonShortley end 
struct B_FHIaims end

_basis_symbols = Dict(:cSH => B_cSH(), 
                      :SpheriCart => B_SpheriCart(), 
                      real => B_SpheriCart(),
                      complex => B_cSH(),
                      B_cSH() => B_cSH(),
                      B_SpheriCart() => B_SpheriCart() )


# ------------------------------------------------------- 

#  NOTE: Ctran(L) is the transformation matrix from rSH to cSH. More specifically, 
#        if we write Polynomials4ML rSH as R_{lm} and cSH as Y_{lm} and their 
#        corresponding vectors of order L as R_L and Y_L, respectively. 
#        Then R_L = Ctran(L) * Y_L. This suggests that the "D-matrix" for the 
#        Polynomials4ML rSH is Ctran(l) * D(l) * Ctran(L)', where D, the 
#        D-matrix for cSH. This inspires the following new CG recursion.

# transformation matrix from RSH to CSH for different conventions

Ctran(i::Integer, j::Integer, basis::B_cSH) = (i==j)

function Ctran(i::Integer, j::Integer, 
               basis::Union{B_SpheriCart, B_CondonShortley, B_FHIaims})
	
    _order(::B_SpheriCart) = [1,2,3,4]
    _order(::B_CondonShortley) = [4,3,2,1]
    _order(::B_FHIaims) = [4,2,3,1]
    
    order = _order(basis)
	val_list = ComplexF64[(-1)^(i), im, (-1)^(i+1)*im, 1] ./ sqrt(2)
	if abs(i) != abs(j)
		return zero(ComplexF64)
	elseif i == j == 0
		return one(ComplexF64)
	elseif i > 0 && j > 0
		return val_list[order[1]]
	elseif i < 0 && j < 0
		return val_list[order[2]]
	elseif i < 0 && j > 0
		return val_list[order[3]]
    end 
    @assert i > 0 && j < 0
    return val_list[order[4]]
end

Ctran(l::Integer; basis = B_SpheriCart()) = dropzeros(sparse(
    Matrix{ComplexF64}([ Ctran(m, μ, basis) 
                         for m = -l:l, μ = -l:l ])))


# -----------------------------------------------------

# The generalized Clebsch Gordan Coefficients; variables of this function are 
# fully inherited from the first ACE paper. 
function GCG(ll::NTuple{N, T}, mm::NTuple{N, T}, LL::NTuple{N, T},
             M_N::T, basis::B_cSH) where {N, T}

    # @assert -L[N] ≤ M_N ≤ L[N] 
    if (m_filter(mm, M_N, basis) == false) || (LL[1] < abs(mm[1]))
        return 0.0
    end

    M = mm[1]
    C = 1.0
    for k in 2:N
        if LL[k] < abs(M + mm[k])
            return 0.0
        else
            cg = PartialWaveFunctions.clebschgordan(
                        LL[k-1], M, ll[k], mm[k], LL[k], M + mm[k])
            C *= cg
            M += mm[k]
        end
    end

    return C
end


function GCG(ll::NTuple{N, T}, mm::NTuple{N, T}, LL::NTuple{N, T},
             M_N::T, basis::B_SpheriCart) where {N, T}

    if m_filter(mm, M_N, basis) == false || LL[1] < abs(mm[1])
       return 0.0
    end

    C = zero(ComplexF64)
    for M in signed_m(M_N)
        ext_mset = filter( x -> sum(x) == M, signed_mmset(mm) )
        
        for mm1 in ext_mset
            # @assert sum(mm1) == M
            C_loc = GCG(ll, mm1, LL, M, B_cSH())
            coeff = Ctran(M_N, M, basis)' * 
                        prod( Ctran(mm[i], mm1[i], basis) for i in 1:N )
            C_loc *= coeff
            C += C_loc
        end
    end

    # We actually expect real values 
    if abs(C - real(C)) > 1e-10
        error("GCG coefficient is not real: $C")
    end

    return real(C) 
end


# Only when M_N = sum(m) can the CG coefficient be non-zero, so when missing M_N, 
# we return either 
# (1) the full CG coefficient given l, m and L, as a rank 1 vector; 
# (2) or the only one element that can possibly be non-zero on the above vector.
# I suspect that the first option will not be used anyhow, but I keep it for now.
function GCG(l::SVector{N,Int64}, m::SVector{N,Int64}, L::SVector{N,Int64};
             vectorize::Bool=true, basis = B_cSH()) where N 
    if typeof(basis) == B_cSH
        return ( vectorize ? (GCG(l,m,L,sum(m); basis = basis) * 
                                Float64.(I(2L[N]+1)[sum(m)+L[N]+1,:])) 
                           : GCG(l,m,L,sum(m); basis = basis) )
    elseif typeof(basis) == B_SpheriCart
        if vectorize == false && L[N] != 0
            error("""For the rSH basis, the CG coefficient is always a vector 
                     except for the case of L=0.""")
        else
            return (L[N] == 0 ? GCG(l,m,L,L[N]; basis = basis) 
                              : SA[[ GCG(l,m,L,M_N; basis = basis) 
                                     for M_N in -L[N]:L[N] ]...]  )
        end
    else
        error("unknown basis type: $basis")
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

function submset(lmax, lth)
    # lmax stands for the l value of the subsection while lth is the length of 
    # this subsection
    if lth == 1
        return [[l] for l in -lmax:lmax]
    else
        tmp = submset(lmax, lth-1)
        mset = Vector{Int64}[]
        for t in tmp
            set = identity.([[t..., l] for l in t[end]:lmax])
            push!(mset, set...)
        end
    end
    return mset
end

# The set of integers that has the same absolute value as m
signed_m(m::T) where {T} = unique([-m, m])::Vector{T}   

# The set of vectors whose i-th element has the same absolute value as m[i] 
# for all i
# The following implementation is elegant but inherentaly type-unstable 
# old_signed_mmset(mm) = Iterators.product([signed_m(m) for m in mm]...
#                                     ) |> collect 
# the tuple implementation is stable, but the product still has overhead. 
# old_signed_mmset2(mm) = 
#         Iterators.product(ntuple(i -> signed_m(mm[i]), length(mm))...)
# the following implementation is a bit cryptic but fairly efficient. 

function signed_mmset(mm::NTuple{N, T}, prune = true) where {N, T}
    len = 2^N
    MM = Vector{NTuple{N, T}}(undef, len)
    σ = zeros(Bool, N)
    for i in 1:(2^N)
       digits!(σ, i-1, base=2)
       newmm = ntuple(j -> ((2*σ[j]-1) * (mm[j] != 0) + (mm[j] == 0)) * mm[j], N)
       MM[i] = newmm
    end
    if prune 
        return unique(MM)
    else 
        return MM 
    end
 end
 

function m_filter(mm, k::Integer, basis::B_cSH)
    return sum(mm) == k
end

function m_filter(mm, k::Integer, basis::B_SpheriCart)
    # no need to make mmset unique, repetitions are not a problem here 
    mmset = signed_mmset(mm, false) 
    for mm1 in mmset
        if sum(mm1) == k
            return true
        end
    end
    return false
end

# Function that generates the set of ordered m's given `n` and `l` with sum of 
# m's equaling to k.
#
# NB: functino assumes lexicographical ordering
#
function m_generate(n::T,l::T,L,k; basis = B_cSH() ) where T
    @assert abs(k) ≤ L
    S = Sn(n,l)
    Nperm = length(S)-1
    ordered_mset = [submset(l[S[i]], S[i+1]-S[i]) for i = 1:Nperm]
    MM = []
    Total_length = 0
    for m_ord in Iterators.product(ordered_mset...)
        m_ord_reshape = vcat(m_ord...)
        if m_filter(m_ord_reshape, k, basis)
            class_m = vcat( Iterators.product( 
                            [ multiset_permutations(m_ord[i], S[i+1]-S[i]) 
                              for i in 1:Nperm]...)... )
            push!(MM, [vcat(mm...) for mm in class_m])
            Total_length += length(class_m)
        end
    end
    return [ T.(MM[i]) for i = 1:length(MM) ], Total_length
end

# Function that generates the set of ordered m's given `n` and `l` with the 
# absolute sum of m's being smaller than L.
# orginal version: sum(m_generate(n,l,L,k;flag)[2] for k in -L:L), 
#                  but this cannot be true anymore b.c. the m_classes can 
#                  intersect
m_generate(n,l,L; basis = B_cSH() ) = 
        union([m_generate(n,l,L,k; basis = basis)[1] for k in -L:L]...), 
        sum(length.(union([m_generate(n,l,L,k; basis = basis)[1] for k in -L:L]...)))

function gram(X::Matrix{SVector{N, T}}) where {N, T}
    G = zeros(T, size(X,1), size(X,1))
    for i = 1:size(X,1)
       for j = i:size(X,1)
          G[i,j] = sum(dot(X[i,t], X[j,t]) for t = 1:size(X,2))
          i == j ? nothing : (G[j,i]=G[i,j]')
       end
    end
    return G
end

gram(X::Matrix{<: Number}) = X * X'

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
        _ll = NTuple{N, Int}(ll...)
    catch 
        error("""coupling_coeffs(L::Integer, ll, ...) requires ll to be 
               a vector or tuple of integers""")
    end

    # convert nn into an SVector{N, Int}, as required internally 
    if isnothing(nn) 
        if PI 
            _nn = ntuple(i -> 0, N)
        else 
            _nn = ntuple(i -> i, N)
        end
    elseif length(nn) != N 
        error("""coupling_coeffs(L::Integer, ll, nn) requires ll and nn to be 
               of the same length""")
    else
        _nn = try 
            _nn = NTuple{N, Int}(nn...)
        catch 
            error("""coupling_coeffs(L::Integer, ll, nn) requires nn to be 
                   a vector or tuple of integers""")
        end
    end 
    
    return _coupling_coeffs(_L, _ll, _nn; 
                            PI = PI, basis = _basis_symbols[basis])
end
    

# Function that generates the coupling coefficient of the RE basis (PI = false) 
# or RPE basis (PI = true) given `nn` and `ll`. 
function _coupling_coeffs(L::Int, ll::NTuple{N, Int}, nn::NTuple{N, Int}; 
                          PI = true, basis) where N

    # NOTE: because of the use of m_generate, the input (nn, ll ) is required
    # to be in lexicographical order.
    nn, ll, inv_perm = lexi_ord(nn, ll)

    Lset = SetLl(ll,L)
    r = length(Lset)
    T = L == 0 ? Float64 : SVector{2L+1,Float64}
    if r == 0 
        return zeros(T, 0, 0), SVector{N, Int}[]
    else 
        MMmat, size_m = m_generate(nn,ll,L; basis = basis) # classes of m's
        FMatrix=zeros(T, r, length(MMmat)) # Matrix containing f(m,i)
        UMatrix=zeros(T, r, size_m) # Matrix containing the the coupling coefs D
        MM = SVector{N, Int}[] # all possible m's
        for i in 1:r
            c = 0
            for (j,m_class) in enumerate(MMmat)
                for mm in m_class
                    c += 1
                    cg_coef = GCG(ll,mm,Lset[i];vectorize=(L!=0), basis = basis)
                    FMatrix[i,j]+= cg_coef
                    UMatrix[i,c] = cg_coef
                end
            end
            @assert c==size_m
        end 
        for m_class in MMmat
            for mm in m_class
                push!(MM, mm)
            end
        end      
    end

    if !PI
        # return RE coupling coeffs if the permutation invariance is not needed
        return UMatrix, [mm[inv_perm] for mm in MM] # MM
    else
        U, S, V = svd(gram(FMatrix))
        # Somehow rank is not working properly here, might be a relative  
        # tolerance issue.
        # original code: rank(Diagonal(S); rtol =  1e-12) 
        rk = findall(x -> x > 1e-12, S) |> length 
        # return the RE-PI coupling coeffs
        return Diagonal(S[1:rk]) * (U[:, 1:rk]' * UMatrix), 
               [ mm[inv_perm] for mm in MM ]
    end
end

end
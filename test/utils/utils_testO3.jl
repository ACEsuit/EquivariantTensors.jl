
const ___UTILS_FOR_TESTS___ = true 

using Test, SpheriCart, StaticArrays, BlockDiagonals
using SpheriCart: idx2lm, lm2idx 
using EquivariantTensors.O3: Ctran

##

using ACEbase.Testing: println_slim, print_tf

# print_tf(::Test.Pass) = printstyled("+", bold=true, color=:green)
# print_tf(::Test.Fail) = printstyled("-", bold=true, color=:red)
# print_tf(::Tuple{Test.Error,Bool}) = printstyled("x", bold=true, color=:magenta)

# println_slim(::Test.Pass) = printstyled("Test Passed\n", bold=true, color=:green)
# println_slim(::Test.Fail) = printstyled("Test Failed\n", bold=true, color=:red)


##

# A naive implementation of a transformed chebyshev basis purely for 
# testing; assumes that norm(𝐫) ≤ 1.
#
function eval_cheb(𝐫::AbstractVector, nmax)
   r = norm(𝐫)
   x = (0.1 + r) / 1.2
   return [ cos( (n-1) * acos(x) ) for n = 1:nmax ]
end 

# evaluate the real spherical harmonics via SpheriCart
function rYlm(L::Integer, 𝐫::SVector{3, T}; convention = :SpheriCart) where {T}
   if convention == :SpheriCart 
      basis = SpheriCart.SphericalHarmonics(L)
   else
      @error("Only SpheriCart convention is supported for now.")
   end
   return Vector(basis(𝐫))
end


# evaluate the complex spherical harmonics; first real via SpheriCart, and then 
# the transformation to complex. 
function cYlm(L::Integer, 𝐫::SVector{3, T}; method = :LC, convention = :SpheriCart) where {T} 
   basis = SpheriCart.SphericalHarmonics(L)
   Y = Vector(Complex{T}.(basis(𝐫)))
   _convert_R2C!(Y, L; method=method, convention=convention)
   return Y 
end


# convert real to complex SH; 
# TODO: can this be replaced by Ctran? 
# LZ: Yes, I will add an option of using Ctran and compare
#     I would make the linear combination approach the default one, though, 
#     because if we use Ctran, and test with Ctran, that would more or less be 
#     a petitio principii
# m = 0 => do nothing; m ≠ 0 => linear combinations of ± m terms 
function _convert_R2C!(Y::AbstractVector, LMAX::Integer; method = :LC, convention = :SpheriCart) # method can be :Ctran or :LC (Linear Combination)
   @assert length(Y) == (LMAX+1)^2
   if method == :LC
      for l = 0:LMAX, m = 1:l 
         i_lm⁺ = lm2idx(l,  m)
         i_lm⁻ = lm2idx(l, -m)
         Ylm⁺ = Y[i_lm⁺]
         Ylm⁻ = Y[i_lm⁻]
         Y[i_lm⁺] = (-1)^m * (Ylm⁺ + im * Ylm⁻) / sqrt(2)
         Y[i_lm⁻] = (Ylm⁺ - im * Ylm⁻) / sqrt(2)
      end
   elseif method == :Ctran
      Y .= BlockDiagonal([ Ctran(L;convention) for L = 0:LMAX ])' * Y
   end
	return Y 
end 


##

# generate a random point on the unit sphere
function rand_sphere() 
   u = @SVector randn(3)
   return u / norm(u) 
end

# generate a random point in the unit ball, NB not uniformly distributed, 
# the density of points is higher near the center of the ball.
rand_ball() = rand_sphere() * rand()

# generate a random rotation matrix 
function rand_rot() 
   K = @SMatrix randn(3,3)
   return exp(K - K') 
end

# generate a random configuration of nX points in the unit ball
rand_config(nX::Integer) = [ rand_ball() for _ in 1:nX ]

# generate a random configuration of nX points in the unit ball with 
# nX drawn randomly from the range `rg`
rand_config(rg::UnitRange) = rand_config(rand(rg))

# generate a random batch of ntest configurations
make_batch(ntest, nX) = [ rand_config(nX) for _ = 1:ntest ] 

# --------------------------------------------------

# Evaluate all coupled basis functions for the given (ll, nn) block 
# as defined by the coupling coefficients `coeffs` (with `MM` being the 
# list of mm tuples over which we have to do the summation).
# 
function eval_basis(Rs; coeffs, MM, ll, nn, Real = false)
   @assert minimum(nn) >= 1 # radial basis indexing starts at 1 not 0. 
   @assert size(coeffs, 2) == length(MM) 

   # correlation order 
   ORD = length(ll) 
   @assert length(nn) == ORD
   @assert all( length(mm) == ORD for mm in MM )
   @assert length(Rs) == ORD # only for the non-sym basis!!

   # spherical harmonics 
   Lmax = maximum(ll)
   _Ylm = Real ? rYlm : cYlm
   Y = [ _Ylm(Lmax, 𝐫) for 𝐫 in Rs ]

   # radial basis 
   T = [ eval_cheb(𝐫, maximum(nn)) for 𝐫 in Rs ]

   if size(coeffs,1) == 0
      return zeros(valtype(coeffs), 0)
   end
   BB = zeros(typeof(coeffs[1]), size(coeffs, 1))
   for i_mm = 1:length(MM)
      mm = MM[i_mm]
      ii_lm = [ lm2idx(ll[α], mm[α]) for α in 1:ORD ]
      BB += coeffs[:, i_mm] * prod( Y[α][ii_lm[α]] * T[α][nn[α]] for α = 1:ORD )
   end 

   return BB
end
 
# same as `eval_basis` except that we first pool the inputs, this means the 
# output will be permutation-invariant as in ACE.
#
function eval_sym_basis(Rs; coeffs, MM, ll, nn, Real = false)
   @assert minimum(nn) >= 1 # radial basis indexing starts at 1 not 0. 
   @assert size(coeffs, 2) == length(MM) 

   # correlation order 
   ORD = length(ll) 
   @assert length(nn) == ORD
   @assert all( length(mm) == ORD for mm in MM )

   # spherical harmonics 
   Lmax = maximum(ll)
   _Ylm = Real ? rYlm : cYlm
   Y = [ _Ylm(Lmax, 𝐫) for 𝐫 in Rs ]

   # radial basis 
   T = [ eval_cheb(𝐫, maximum(nn)) for 𝐫 in Rs ]

   # pooled tensor product operation -> A[i_lm, n]
   A = sum( Y[j] * T[j]' for j = 1:length(Rs) )

   BB = zeros(typeof(coeffs[1]), size(coeffs, 1))
   for i_mm = 1:length(MM)
      mm = MM[i_mm]
      ii_lm = [ lm2idx(ll[α], mm[α]) for α in 1:ORD ]
      BB += coeffs[:, i_mm] * prod( A[ii_lm[α], nn[α]] for α = 1:ORD )
   end 

   return BB
end


# ---------------------------------------------------------------------------
# CO: It's unclear to me why any of the below are needed, some 
#     comments and documentation would be very nice. 

function rand_batch(; coeffs, MM, ll, nn, 
                     ntest = 100, 
                     batch = make_batch(ntest, length(ll)), Real = false) 
   if size(coeffs,1) == 0
      return zeros(valtype(coeffs), 0, length(batch))
   end
   BB = complex.(zeros(typeof(coeffs[1]), size(coeffs, 1), length(batch)))
   for (i, Rs) in enumerate(batch)
      BB[:, i] = eval_basis(Rs; coeffs=coeffs, MM=MM, ll=ll, nn=nn, Real=Real) 
   end
   return BB
end

function sym_rand_batch(; coeffs, MM, ll, nn, 
                        ntest = 100, 
                        batch = make_batch(ntest, length(ll)), Real = false) 
   if size(coeffs,1) == 0
      return BB = zeros(valtype(coeffs), 0, length(batch))
   end
   BB = complex.(zeros(typeof(coeffs[1]), size(coeffs, 1), length(batch)))
   for (i, Rs) in enumerate(batch)
      BB[:, i] = eval_sym_basis(Rs; coeffs=coeffs, MM=MM, ll=ll, nn=nn, Real=Real)
   end
   return BB
end

# # The following two functions are hacked from the EQM package, just using as reference and for comparison
# # they will be moved to EquivariantTensors soon
# function rpe_basis(A::Union{Rot3DCoeffs,Rot3DCoeffs_real}, nn::SVector{N, TN}, ll::SVector{N, Int}) where {N, TN}
#    t_re_old = @elapsed Ure, Mre = re_basis(A, ll)
#    # @show t_re_old
#    G = _gramian(nn, ll, Ure, Mre)
#    S = svd(G)
#    rk = rank(Diagonal(S.S); rtol =  1e-7)
#    Urpe = S.U[:, 1:rk]'
#    return Diagonal(sqrt.(S.S[1:rk])) * Urpe * Ure, Mre
# end


# function _gramian(nn, ll, Ure, Mre)
#    N = length(nn)
#    nre = size(Ure, 1)
#    G = zeros(Complex{Float64}, nre, nre)
#    for σ in permutations(1:N)
#       if (nn[σ] != nn) || (ll[σ] != ll); continue; end
#       for (iU1, mm1) in enumerate(Mre), (iU2, mm2) in enumerate(Mre)
#          if mm1[σ] == mm2
#             for i1 = 1:nre, i2 = 1:nre
#                G[i1, i2] += coco_dot(Ure[i1, iU1], Ure[i2, iU2])
#             end
#          end
#       end
#    end
#    return G
# end

##

index_y(l::Integer, m::Integer) = m + l + (l*l) + 1 # function hacked from P4ML

function eval_basis(ll, Ure, Mll, X; Real = true)
    @assert length(X) == length(ll)
    @assert all(length.(X) .== 3) 
 
    # NOTE: It seems that we will not go beyond vector valued functions in this package...?
    _convert = complex # Real ? real : complex # identity
    val = _convert(zeros(typeof(Ure[1]), size(Ure,1)))
    
    Lmax = maximum(ll)
    _Ylm = Real ? rYlm : cYlm
    Ylm = [ _Ylm(Lmax, 𝐫) for 𝐫 in X ]
 
    for (i, mm) in enumerate(Mll)
       prod_Ylm = prod( Ylm[j][index_y(l, m)] 
                        for (j, (l, m)) in enumerate(zip(ll, mm)) )
       val .+= _convert(Ure[:,i] * prod_Ylm)
    end
 
    return val 
end
 

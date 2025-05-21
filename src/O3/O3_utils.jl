using Rotations
import PartialWaveFunctions, WignerD 

#  NOTE: Ctran(L) is the transformation matrix from rSH to cSH. More specifically, 
#        if we write Polynomials4ML rSH as R_{lm} and cSH as Y_{lm} and their 
#        corresponding vectors of order L as R_L and Y_L, respectively. 
#        Then R_L = Ctran(L) * Y_L. This suggests that the "D-matrix" for the 
#        Polynomials4ML rSH is Ctran(l) * D(l) * Ctran(L)', where D, the 
#        D-matrix for cSH.

# transformation matrix from RSH to CSH for different conventions

# NOTE: removing this dictionary for now since we decided to allow only the 
#       real and complex sphericart basis 
#
# const order_dict = Dict(:SpheriCart => SA[1,2,3,4], 
#                       :CondonShortley => SA[4,3,2,1], 
#                       :FHIaims => SA[4,2,3,1] )

_Ctran(i::Integer, j::Integer) = _Ctran(i, j, real)

_Ctran(i::Integer, j::Integer, basis::typeof(complex)) = (i == j)


function _Ctran(i::Integer, j::Integer, basis::typeof(real))
	val_list = SA[(-1)^(i), im, (-1)^(i+1)*im, 1] / sqrt(2)
   T = eltype(val_list)
	if abs(i) != abs(j)
		return zero(T)
	elseif i == j == 0
		return one(T)
	elseif i > 0 && j > 0
		return val_list[1]
	elseif i < 0 && j < 0
		return val_list[2]
	elseif i < 0 && j > 0
		return val_list[3]
   end
   @assert i > 0 && j < 0
	return val_list[4]
end


Ctran(l::Int64; basis = real) = 
         sparse([ _Ctran(m, μ, basis) for m = -l:l, μ = -l:l ])


Ctran(mm1::SVector{N,Int}, mm2::SVector{N,Int}, basis = real) where {N} = 
      ( abs.(mm1) == abs.(mm2) 
         ? conj(prod(_Ctran(mm2[t], mm1[t], basis) for t = 1:N))
         : zero(ComplexF64) )::ComplexF64


Ctran(mm1::Vector{Int}, mm2::Vector{Int}, basis = real) = 
      ( abs.(mm1) == abs.(mm2) 
        ? prod(_Ctran(mm2[i], mm1[i], basis)' for i in 1:length(mm1))
        : zero(ComplexF64) )


# We also need to define the transformation matrix from product rSH to product cSH

# grouping those MM's that has the same abs value
function group_by_abs(MM::Vector{SVector{N,Int}}) where N
   abs_map = Dict{NTuple{N, Int}, Vector{Int}}()
   for (idx, v) in enumerate(MM)
       key = Tuple(abs.(v))  # use tuple as a hashable key
       push!(get!(abs_map, key, Int[]), idx)
   end
   return abs_map
end

function rAA2cAA(MM_c, MM_r; basis = real)
   # find the abs.(mm) and group
   group_c = group_by_abs(MM_c)
   group_r = group_by_abs(MM_r)

   # Match groups and fill sparse matrix accordingly
   # TODO: replace with triplet format and then use sparse constructor
   CC = spzeros(ComplexF64, length(MM_c), length(MM_r))

   # By the following, we don't need nested loops
   for (key, c_inds) in group_c
       if haskey(group_r, key)
           r_inds = group_r[key]
           for i in c_inds
               for j in r_inds
                   CC[i, j] = Ctran(MM_c[i], MM_r[j], basis)
               end
           end
       end
   end

   return CC
end


# -----------------------------------------------------------------
#  complex and real Clebsch-Gordan coefficients

function cg(l1, m1, l2, m2, L, M, basis::typeof(complex))
   return PartialWaveFunctions.clebschgordan(l1, m1, l2, m2, L, M)
end

function cg(l1, m1, l2, m2, L, M, basis::typeof(real))
   return _real_clebschgordan(l1, m1, l2, m2, L, M) 
end
 
function _real_clebschgordan(l1, m1, l2, m2, λ, νp)
   result = 0.0 + 0im

   # Only n values with |n| = |m| contribute due to Ctran(m, n) selection rule
   n1_values = m1 == 0 ? [0] : [-m1, m1]
   n2_values = m2 == 0 ? [0] : [-m2, m2]

   for n1 in n1_values
       for n2 in n2_values
           ν = n1 + n2
           # Selection rules: |ν| must be ≤ λ and match |νp|
           if abs(ν) ≤ λ && abs(ν) == abs(νp)
               cg = PartialWaveFunctions.clebschgordan(l1, n1, l2, n2, λ, ν)
               result += ( Ctran(m1, n1) * conj(Ctran(-m2, -n2))
                           * conj(Ctran(νp, ν, basis)) * cg )
           end
       end
   end

   return real(result)
end
 

function cgmatrix(l1, l2, λ; basis = real)
   cgm = zeros((2l1+1)*(2l2+1), 2λ+1)
   for (i, (m1, m2)) in enumerate(Iterators.product(-l1:l1, l2:-1:-l2))
      for ν = -λ : λ
         if abs(ν) <= λ
            cgm[i, ν + λ + 1] = (-1)^m2 * cg(l1, m1, l2, m2, λ, ν, basis)
         end
      end
   end
   return sparse(cgm) 
end


# -----------------------------------------------------------------
#  complex and real D matrices 

function Q_from_angles(θ::AbstractVector{<: Real})
   @assert length(θ) == 3
   return Rotations.RotZYZ(θ...)
end

function D_from_angles(l::Integer, θ::AbstractVector{<: Real}, ::typeof(complex))
   @assert length(θ) == 3
   return conj.(WignerD.wignerD(l, θ...))
end

"""
   D_from_angles(l, θ, basis)

Here, `l::Integer` and `θ` a 3-element vector or tuple, `basis` must be either 
`real` or `complex`. Output is a Wigner-D matrix such that `y ∘ Q = D * y` 
with `y` real/complex spherical harmonics. 
"""
function D_from_angles(l::Integer, θ::AbstractVector{<: Real}, ::typeof(real))
   @assert length(θ) == 3
   cD = WignerD.wignerD(l, θ...)
   T = Ctran(l)
   return real.(T * conj.( cD ) * T')
end

"""
produces a rotation Q and Wigner-D matrix D such that `y ∘ Q = D * y` with `y`
real spherical harmonics. 
"""
QD_from_angles(l::Integer, θ::AbstractVector{<: Real}, RC) = 
         Q_from_angles(θ), D_from_angles(l, θ, RC)

using Rotations
import WignerD 

#  NOTE: Ctran(L) is the transformation matrix from rSH to cSH. More specifically, 
#        if we write Polynomials4ML rSH as R_{lm} and cSH as Y_{lm} and their 
#        corresponding vectors of order L as R_L and Y_L, respectively. 
#        Then R_L = Ctran(L) * Y_L. This suggests that the "D-matrix" for the 
#        Polynomials4ML rSH is Ctran(l) * D(l) * Ctran(L)', where D, the 
#        D-matrix for cSH.

# transformation matrix from RSH to CSH for different conventions

function Ctran(i::Int64,j::Int64;convention = :SpheriCart)
	if convention == :cSH
		return i == j
	end
	
	order_dict = Dict(:SpheriCart => [1,2,3,4], 
                      :CondonShortley => [4,3,2,1], 
                      :FHIaims => [4,2,3,1] )

	val_list = [(-1)^(i), im, (-1)^(i+1)*im, 1] ./ sqrt(2)
	if abs(i) != abs(j)
		return 0 
	elseif i == j == 0
		return 1
	elseif i > 0 && j > 0
		return val_list[order_dict[convention][1]]
	elseif i < 0 && j < 0
		return val_list[order_dict[convention][2]]
	elseif i < 0 && j > 0
		return val_list[order_dict[convention][3]]
	elseif i > 0 && j < 0
		return val_list[order_dict[convention][4]]
	end
end

Ctran(l::Int64; convention = :SpheriCart) = sparse(
    Matrix{ComplexF64}([ Ctran(m,μ;convention=convention) 
                         for m = -l:l, μ = -l:l ]))

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

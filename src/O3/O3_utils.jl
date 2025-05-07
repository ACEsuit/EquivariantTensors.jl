using Rotations
import PartialWaveFunctions, WignerD 

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
#  complex and real Clebsch-Gordan coefficients

cg_m_condition(m1, m2, M) = (M == m1 + m2)

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
               result += Ctran(m1, n1) * conj(Ctran(-m2, -n2)) * conj(Ctran(νp, ν)) * cg
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
   return real.( conj.( WignerD.wignerD(l, θ...) ) )
end

function D_from_angles(l::Integer, θ::AbstractVector{<: Real}, ::typeof(real))
   @assert length(θ) == 3
   cD = WignerD.wignerD(l, θ...)
   T = Ctran(l)
   return real.(T * conj.( cD ) * T')
end

QD_from_angles(l::Integer, θ::AbstractVector{<: Real}, RC) = 
         Q_from_angles(θ), D_from_angles(l, θ, RC)

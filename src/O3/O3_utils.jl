

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
   C1 = Ctran(l1)
   C2 = Ctran(l2)
   Cλ = Ctran(λ)
 
   val = 0.0 + 0im
   for n1 = -l1:l1
      for n2 = -l2:l2
         ν = n1 + n2
         if abs(ν) <= λ
            cg = PartialWaveFunctions.clebschgordan(l1, n1, l2, n2, λ, ν)
 
            i1 = m1 + l1 + 1
            i2 = (l2 - m2) + 1  
            j1 = n1 + l1 + 1
            j2 = (l2 - n2) + 1 
            k1 = νp + λ + 1
            k2 = ν + λ + 1
 
            val += C1[i1, j1] * conj(C2[i2, j2]) * conj(Cλ[k1, k2]) * cg
         end
      end
   end
   return real(val)
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

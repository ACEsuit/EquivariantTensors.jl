using PartialWaveFunctions
 
function trans_y_pp(y0, y2, params; basis = real) 
   l1 = 1
   l2 = 1
   y0 = SVector{1}(y0)
   return _block((y0, y2), l1, l2, params; basis = basis)
end

function _block(x, l1, l2, params; basis = real)
   fea_set = abs(l1-l2):2:l1+l2
   cgm = [cgmatrix(l1, l2, λ; basis = basis) for λ in fea_set]
   H = reshape.(cgm .* sparse.(x), 2l1+1, 2l2+1)
   return sum(params .* H)
end
 
function cgmatrix(l1, l2, λ; basis = real)
   cgm = zeros((2l1+1)*(2l2+1), 2λ+1)
   for (i, (m1, m2)) in enumerate(Iterators.product(-l1:l1, l2:-1:-l2))
      for ν = -λ : λ
         if abs(ν) <= λ
            cgm[i, ν + λ + 1] = (-1)^m2 * cg(l1, m1, l2, m2, λ, ν; basis = basis)
         end
      end
   end
   return sparse(cgm) 
end
 
cg_m_condition(m1, m2, M) = (M == m1 + m2)
 
function cg(l1, m1, l2, m2, λ, ν; basis = real) 
   if basis == complex
      if !cg_m_condition(m1, m2, ν)
         return 0.0
      else
         return PartialWaveFunctions.clebschgordan(l1, m1, l2, m2, λ, ν) 
      end
   elseif basis == real
      return real_clebschgordan(l1, m1, l2, m2, λ, ν) 
   end
end
 
function real_clebschgordan(l1, m1, l2, m2, λ, νp)
   C1 = ET.O3.Ctran(l1)
   C2 = ET.O3.Ctran(l2)
   Cλ = ET.O3.Ctran(λ)
 
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
 
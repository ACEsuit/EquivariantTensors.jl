
# TODO: these transformations should use static sparse arrays: 
#       https://github.com/QuantumBFS/LuxurySparse.jl/blob/master/src/SSparseMatrixCSC.jl

struct TYVec2pp{T0, T2}
   cgm0::T0 
   cgm2::T2
end

function TYVec2pp(; basis = real)
   cgms = _get_cgms1(1, 1; basis = basis)
   return TYVec2pp(cgms[1], cgms[2])
end 

(t::TYVec2pp)(y0, y2) = SMatrix{3,3}(t.cgm0 * y0 + t.cgm2 * y2)


function _get_cgms1(l1, l2; basis = real)
   fea_set = abs(l1-l2):2:l1+l2
   return cgm = [ cgmatrix(l1, l2, λ; basis = basis) for λ in fea_set ]
end

# ------------------------------------------------------

struct TYVec2YMat{L1, L2, TT}
   cgm::TT 
end

function TYVec2YMat(L1, L2; basis = real)
   cgms = [ cgmatrix(L1, L2, l; basis = basis) for l = 0:(L1+L2) ]
   combined_cg = sparse( hcat(cgms...) )
   TT = typeof(combined_cg)
   return TYVec2YMat{L1, L2, TT}(combined_cg)
end 

function (t::TYVec2YMat{L1, L2})(y::SYYVector{L}) where {L1, L2, L} 
   if L != L1 + L2  
      error("L must be equal to L1 + L2")
   end 
   SMatrix{2*L1+1, 2*L2+1}(t.cgm * y)
end



# function _block(x, l1, l2; basis = real)
#    fea_set = abs(l1-l2):2:l1+l2
#    cgm = [O3.cgmatrix(l1, l2, λ; basis = basis) for λ in fea_set]
#    H = reshape.(cgm .* sparse.(x), 2l1+1, 2l2+1)
#    return sum(H)
# end

# ------------------------------------------------------
#=
using SparseArrays
import EquivariantTensors: O3
 
function trans_y_pp(y0, y2; basis = real) 
   l1 = 1
   l2 = 1
   y0 = SVector{1}(y0)
   return _block((y0, y2), l1, l2; basis = basis)
end

function _block(x, l1, l2; basis = real)
   fea_set = abs(l1-l2):2:l1+l2
   cgm = [O3.cgmatrix(l1, l2, λ; basis = basis) for λ in fea_set]
   H = reshape.(cgm .* sparse.(x), 2l1+1, 2l2+1)
   return sum(H)
end
 
=#
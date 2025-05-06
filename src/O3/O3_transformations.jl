


struct TYVec2pp{T0, T2}
   cgm0::T0 
   cgm2::T2
end

function TYVec2pp(; basis = real) 
   cgms = _get_cgms(1, 1; basis = basis)
   return TYVec2pp(cgms[1], cgms[2])
end 

(t::TYVec2pp)(y0, y2) = SMatrix{3,3}(t.cgm0 * y0 + t.cgm2 * y2)


function _get_cgms(l1, l2; basis = real)
   fea_set = abs(l1-l2):2:l1+l2
   return cgm = [ O3.cgmatrix(l1, l2, λ; basis = basis) for λ in fea_set ]
end


# function _block(x, l1, l2; basis = real)
#    fea_set = abs(l1-l2):2:l1+l2
#    cgm = [O3.cgmatrix(l1, l2, λ; basis = basis) for λ in fea_set]
#    H = reshape.(cgm .* sparse.(x), 2l1+1, 2l2+1)
#    return sum(H)
# end





# struct TYVec2YMat{L1, L2}
#    cgm::TT 
# end


# (t::TYVec2YMat{L1, L2})(y) where {L1, L2} = 
#       reshape(t.cgm * y, 2*L1+1, 2*L2+1) 



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
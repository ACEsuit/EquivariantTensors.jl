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
 


# TODO: these transformations should use static sparse arrays: 
#       https://github.com/QuantumBFS/LuxurySparse.jl/blob/master/src/SSparseMatrixCSC.jl

"""
   struct TYVec2YMat

transformation from a real Y vector to a spherical matrix       
"""
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
   if L < L1 + L2  
      error("L must be equal to L1 + L2 (file an issue if L > L1+L2 is needed)")
   end 
   return SMatrix{2*L1+1, 2*L2+1}(t.cgm * y)
end


""" 
   struct TYVec2CartVec 

transformation from a real Y vector to a cartesian vector; accepts as input 
either an `SVector{3}`` (i.e. the Y_1^m) or a `SYYVector` (containing both 
the Y_0^0 and Y_1^m); the output is a `SVector{3}`.
"""
struct TYVec2CartVec
end

TYVec2CartVec(basis::typeof(real)) = TYVec2CartVec()

TYVec2CartVec(basis::typeof(complex)) = error("Complex basis not supported for TYVec2CartVec")

function (t::TYVec2CartVec)(y::SYYVector{L}) where {L} 
   if L < 1
      @show L 
      error("L must be at least 1")
   end
   return SVector(y[(1,1)], y[(1,-1)], y[(1,0)])
end

function (t::TYVec2CartVec)(y::SVector{3})
   return SVector(y[3], y[1], y[2])
end

function (t::TYVec2CartVec)(y::AbstractVector)
   @assert length(y) == 3 
   return SVector(y[3], y[1], y[2])
end


""" 
   struct TYVec2CartMat 

transformation from a real Y vector to a cartesian matrix; accepts as input 
either an `SVector{3}`` (i.e. the Y_1^m) or a `SYYVector` (containing both 
the Y_0^0 and Y_1^m); the output is a `SMatrix{3,3}`.
"""
struct TYVec2CartMat{TT}
   ty::TYVec2YMat{1, 1, TT}
end

TYVec2CartMat(basis::typeof(real)) = TYVec2CartMat(TYVec2YMat(1, 1; basis = real))

TYVec2CartMat(basis::typeof(complex)) = error("Complex basis not supported for TYVec2CartMat")

function (t::TYVec2CartMat)(y::SYYVector)
   Hy = SMatrix{3,3}(t.ty(y))
   P = SMatrix{3,3}(0, 1, 0, 0, 0, 1, 1, 0, 0)
   return P * Hy * P' 
end

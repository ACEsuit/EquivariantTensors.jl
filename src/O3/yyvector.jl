using StaticArrays 

import Base: *, +, complex, real

struct SYYVector{L, N, T}  <: StaticVector{N, T}
   data::NTuple{N, T}
end

SYYVector(data::NTuple{N, T}) where {N, T} = 
         try SYYVector{Int(sqrt(N)-1), N, T}(data) 
         catch 
            error("length of the input should be L^2 for some Int L!") 
         end

# TODO: @boundscheck / @propagate_inbounds
Base.@propagate_inbounds function Base.getindex(y::SYYVector, i::Int)
	@boundscheck checkbounds(y,i)
	return y.data[i] 
end

@inline _lm2i(l, m) = l^2 + m + l + 1
@inline _i2lm(i) = ( ceil(Int,sqrt(i)) - 1, i - ceil(Int,sqrt(i))^2 + ceil(Int,sqrt(i)) - 1)::Tuple{Int,Int}

@inline Base.getindex(y::SYYVector, lm::Tuple{Int,Int}) = y[_lm2i(lm[1], lm[2])]
@inline Base.getindex(y::SYYVector, lm::NamedTuple{(:l,:m),Tuple{Int,Int}}) = y[_lm2i(lm.l, lm.m)]

@inline Base.getindex(y::SYYVector, ::Val{l}) where l = 
      SVector(ntuple(i -> y[i+l^2], 2*l+1))

Base.Tuple(y::SYYVector) = y.data 

*(y::SYYVector{L,N,T1},b::T2) where {L,N,T1<:Number,T2<:Number} = 
      SYYVector(NTuple{N,promote_type(T1,T2)}([ y.data[i] for i = 1:N ] * b))

+(y::SYYVector{L,N,T1},b::T2) where {L,N,T1<:Number,T2<:Number} = 
		SYYVector(NTuple{N,promote_type(T1,T2)}([ y.data[i] for i = 1:N ] .+ b))

+(y1::SYYVector{L,N,T1},y2::SYYVector{L,N,T2}) where {L,N,T1<:Number,T2<:Number} = 
		SYYVector(NTuple{N,promote_type(T1,T2)}([ y1.data[i]+y2.data[i] for i = 1:N ]))
		
complex(y::SYYVector{L,N,T}) where {L,N,T<:Number} = SYYVector{L,N,ComplexF64}(y.data)
real(y::SYYVector{L,N,T}) where {L,N,T<:Number} = 
		try SYYVector{L,N,Float64}(y.data)
		catch
			error("Can not convert a complex SYYVector to a real one.")
		end

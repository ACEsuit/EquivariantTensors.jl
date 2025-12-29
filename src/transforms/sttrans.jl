
"""
Generate a transform with frozen parameters stored in a state. 
The advantage of this layer over WrappedFunction is that the state can be 
moved to a device or converted to different floating point formats. 
"""
function st_transform(f::Function, refstate::NamedTuple)
   TransformST(f, refstate)
end


"""
Generate a transform with frozen parameters stored in a state; cf 
`st_transform`.
"""
struct TransformST{FT, ST} <: AbstractLuxLayer
   f::FT 
   refstate::ST
end

Base.show(io::IO, l::TransformST) = print(io, "TransformST()")

initialparameters(rng::AbstractRNG, l::TransformST) = NamedTuple()
initialstates(rng::AbstractRNG, l::TransformST) = deepcopy(l.refstate)

(l::TransformST)(x, ps, st) = l.f(x, st), st 

# this non-standard calling convention assumes that st is not changed 
(l::TransformST)(x, st) = l.f(x, st)
(l::TransformST)(x) = l.f(x, l.refstate)

(l::TransformST)(x::AbstractVector, ps, st) = l(x, st), st 
(l::TransformST)(x::AbstractVector, st) = broadcast(l.f, x, Ref(st))

evaluate(l::TransformST, x, ps, st) = l.f(x, st)

function evaluate_ed(l::TransformST, x::AbstractVector, ps, st)
   y = broadcast(l.f, x, Ref(st))
   # dy = let l_f = l.f, st = st 
   #    broadcast(DiffNT.grad_fd, Ref(l_f), x, Ref(st)) 
   # end 
   dy = map(xi -> DiffNT.grad_fd(l.f, xi, st), x)
   return (y, dy), st
end
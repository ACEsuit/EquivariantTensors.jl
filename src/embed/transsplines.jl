#
# This is an essentially complete re-impleentation of P4ML spline 
# evaluation  because some small details just don't run nice on GPU 
# TODO : 
#   (1) check whether this code can be unified between P4ML and ET 
#       to reduce duplication 
#   (2) consider parallelizing the KA evaluation over both inputs and 
#       spline output indices if the spline output is a vector 
#       (almost always, maybe always??) 
#

function trans_splines(trans, splines, selector;  
                       envelope = nothing)
   # precompute states for the splines 
   states = [ P4ML._init_luxstate(spl) for spl in splines ]
   FF = reduce(hcat, [ s.F for s in states ])
   GG = reduce(hcat, [ s.G for s in states ])
   X0 = [ s.x0 for s in states ]
   X1 = [ s.x1 for s in states ]
   refstate = (; F = FF, G = GG, x0 = X0, x1 = X1 )
   return TransSelSplines(trans, envelope, selector, refstate)
end 

function trans_splines(embed::EmbedDP, ps, st; 
                       yrange = (-1.0, 1.0), nspl = 100)
   if !(embed.post isa SelectLinL)
      error("auto conversion to splines only supported for post = SelectLinL")
   end

   trans = embed.trans
   WW = ps.post.W 
   NCAT = size(WW, 3)
   splines = [ P4ML.splinify( y -> WW[:, :, i] * embed.basis(y), 
                              yrange[1], yrange[2], nspl ) 
               for i in 1:NCAT ] 
   return trans_splines(trans, splines, embed.post.selector)
end


"""
   struct TransSelSplines

A spline implementation that is intended exclusively to be used as an invariant 
embedding, e.g. radial basis. 
```
x -> y -> s(y, i(x)) ->   R(x) = s(y) .* e(x) 
  ------> e(x) ------/
```
here `y = trans(x)`, the selector chooses the index `i(x)` of the spline, 
`e(x)` is defined by the envelope. 
"""
@concrete struct TransSelSplines  <: AbstractLuxLayer
   trans           # transform 
   envelope        # envelope
   selector        # selector 
   refstate        # reference spline parameters (frozen hence states)
end 

initialstates(rng::AbstractRNG, l::TransSelSplines) = 
      ( trans = LuxCore.initialstates(rng, l.trans), 
        envelope = LuxCore.initialstates(rng, l.envelope), 
      #   selector = LuxCore.initialstates(rng, l.selector), 
        params = deepcopy(l.refstate) )

(l::TransSelSplines)(x, ps, st) = _apply_etsplinebasis(l, x, st), st 

evaluate(l::TransSelSplines, x, ps, st) = _apply_etsplinebasis(l, x, st), st 

      
function _apply_etsplinebasis(l::TransSelSplines, 
                              X::AbstractVector{<: XState}, 
                              st)
   # transform 
   Y = l.trans(X, st.trans) 
   # select the spline parameters 
   i_sel = broadcast(l.selector, X)
   # allocate 
   len = length(eltype(st.params.F))
   S = similar(Y, eltype(Y), (length(X), len))

   backend = KernelAbstractions.get_backend(X)
   KernelAbstractions.synchronize(backend)

   # TODO: merge the envelope into the kernel for efficiency 
   #       (visit each memory  location only once) 

   @kernel function _etspl_kernel!(S, Y, i_sel, FF, GG, X0, X1, NX)
      idx = @index(Global)
      y = Y[idx]
      icat = i_sel[idx]
      x0 = X0[icat]
      x1 = X1[icat]

      x = clamp(y, x0, x1)    # project to [x0, x1] (corresponds to Flat bc)
      h = (x1 - x0) / (NX-1)                 # uniform grid spacing 
      # il = floor(Int, (x - x0) / h)         # index of left node
      il = unsafe_trunc(Int, (x - x0) / h)   # floor doesn't run on Metal 
      # TODO: is this numerically stable? 
      t = (x - x0) / h - il            # relative coordinate of x in [il, il+1]

      s = _eval_cubic(t, FF[il+1, icat], FF[il+2, icat], 
                       h*GG[il+1, icat], h*GG[il+2, icat])

      S[idx, :] .= s

      nothing
   end

   kernel! = _etspl_kernel!(backend)
   kernel!(S, Y, i_sel, st.params.F, st.params.G, st.params.x0, st.params.x1, 
           size(st.params.F, 1); ndrange = (length(X),) )

   if l.envelope != nothing 
      ee, _ = l.envelope(X, NamedTuple(), st.envelope)
      S .= ee .* S
   end 
   
   return S
end


function evaluate_ed(l::TransSelSplines, 
                     X::AbstractVector{<: XState}, 
                     ps, st)
   # transform 
   (Y, dY), _ = evaluate_ed(l.trans, X, NamedTuple(), st.trans)
   # select the spline parameters
   i_sel = broadcast(l.selector, X)
   # allocate
   S = similar(Y, eltype(Y), (length(X), length(l.ref_spl)))
   ∂S = similar(dY, eltype(dY), (length(X), length(l.ref_spl)))

   for (idx, y) in enumerate(Y)
      spl_idx = st.params[i_sel[idx]]
      s_i, ds_i = P4ML.evaluate_ed(l.ref_spl, y, nothing, spl_idx)
      S[idx, :] = s_i
      ∂S[idx, :] = Ref(dY[idx]) .* ds_i
   end

   if l.envelope != nothing 
      (ee, ∂ee), _ = evaluate_ed(l.envelope, X, NamedTuple(), st.envelope)
      ∂S .= ∂ee .* S .+ ee .* ∂S
      S .= ee .* S
   end

   return (S, ∂S), st
end


function rrule(::typeof(_apply_etsplinebasis), 
               l::TransSelSplines, X::AbstractArray, st)

   (P, dP), st = evaluate_ed(l, X, NamedTuple(), st)

   function _pb_etsplinebasis(_∂P)
      ∂P = unthunk(_∂P)
      ∂X = dropdims( sum(∂P .* dP, dims = 2), dims = 2) 
      return NoTangent(), NoTangent(), ∂X, NoTangent(), NoTangent()
   end

   return P, _pb_etsplinebasis
end



"""
   _eval_cubic(t, fl, fr, gl, gr, h)

Evaluate cubic spline at position `t` in `[0,1]`, given function values `fl`, `fr`
and gradients `gl`, `gr` at the left and right endpoints.
"""
@inline function _eval_cubic(t, fl, fr, gl, gr)
   # (2t³ - 3t² + 1)*fl + (t³ - 2t² + t)*gl + 
   #           (-2t³ + 3t²)*fr + (t³ - t²)*gr 
   a0 = fl
   a1 = gl 
   a2 = -3fl + 3fr - 2gl - gr
   a3 = 2fl - 2fr + gl + gr
   return ((a3*t + a2)*t + a1)*t + a0
end

"""
   _eval_cubspl(x, F, G, x0, x1, NX)

auxiliary function to the evaluate the cubic spline basis given 
the spline data arrays    
"""
@inline function _eval_cubspl(x, F, G, x0, x1, NX)
   x = clamp(x, x0, x1)     # project to [x0, x1] (corresponds to Flat bc)
   h = (x1 - x0) / (NX-1)   # uniform grid spacing 
   il = unsafe_trunc(Int, (x - x0) / h)   # index of left node
   # TODO: is this numerically stable? 
   t = (x - x0) / h - il          # relative coordinate of x in [il, il+1]
   @inbounds _eval_cubic(t, F[il+1], F[il+2], h*G[il+1], h*G[il+2])
end

@inline function _cubspl_widthgrad(x, F, G, x0, x1, NX)
   if x < x0 || x > x1
      f = _eval_cubspl(x, F, G, x0, x1, NX)
      return f, zero(f) 
   end
   h = (x1 - x0) / (NX-1)   # uniform grid spacing 
   t, _il = modf((x - x0) / h)
   il = Int(_il)
   td = Dual(t, one(t))
   fd = _eval_cubic(td, F[il+1], F[il+2], h*G[il+1], h*G[il+2])
   f = ForwardDiff.value.(fd)
   g = ForwardDiff.partials.(fd, 1)
   return f, g / h 
end

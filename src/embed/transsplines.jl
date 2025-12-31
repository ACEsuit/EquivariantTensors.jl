

function trans_splines(trans, splines, selector;  
                       envelope = nothing)
   # precompute states for the splines 
   states = [ P4ML._init_luxstate(spl) for spl in splines ]
   return TransSelSplines(trans, envelope, selector, splines[1], states)
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
   ref_spl         # reference spline basis (ignore the stored parameters)
   states          # reference spline parameters (frozen hence states)
end 

initialstates(rng::AbstractRNG, l::TransSelSplines) = 
      ( trans = LuxCore.initialstates(rng, l.trans), 
        envelope = LuxCore.initialstates(rng, l.envelope), 
      #   selector = LuxCore.initialstates(rng, l.selector), 
        params = deepcopy(l.states) )

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
   S = similar(Y, eltype(Y), (length(X), length(l.ref_spl)))

   for (idx, y) in enumerate(Y)
      spl_idx = st.params[i_sel[idx]]
      S[idx, :] = P4ML.evaluate(l.ref_spl, y, nothing, spl_idx)
   end

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
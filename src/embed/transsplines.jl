

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

LuxCore.initialstates(rng::AbstractRNG, l::TransSelSplines) = 
      ( trans = LuxCore.initialstates(rng, l.trans), 
        envelope = LuxCore.initialstates(rng, l.envelope), 
      #   selector = LuxCore.initialstates(rng, l.selector), 
        params = deepcopy(l.states) )

(l::TransSelSplines)(x, ps, st) = _apply_etsplinebasis(l, x, st), st 

evaluate(l::TransSelSplines, x, ps, st) = _apply_etsplinebasis(l, x, st)

      
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
      S1 = ee .* S
      ∂S .= ∂ee .* S .+ ee .* ∂S
   end

   return (S1, ∂S), st
end
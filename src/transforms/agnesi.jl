

"""
   gagnesi_params(pcut, pin, rin, req, rcut)

Precompute the parameters for the generalized Agnesi transform, to be used with 
`eval_gagnesi`. See docs for ??? for details. 
"""
function gagnesi_params(pcut::Integer, pin::Integer, 
                        rin::Real, req::Real, rcut::Real)
   @assert p > 0
   @assert q > 0
   @assert q >= p      
   @assert 0 <= rin < req < rcut

   # compute the parameter that maximizes the slope at r = req 
   a = (-2 * q + p * (-2 + 4 * q)) / (p + p^2 + q + q^2)
   @assert a > 0 
   # this defines the following transform 
   #   (setting rin ≡ 0, p = pcut, q = pin ... old notation)
   # x(r) = \frac{1}{1 + a (r/req)^q / (1 + (r/req)^(q-p))}
   # x(r) ~ (1+ a (r/req)^p)^(-1) ~ (r/req)^{-p} as r -> ∞ 
   # x(r) ~ (1 + a (r/req)^q)^(-1) ~ 1 - a (r/req)^q as r -> 0
   _s(r) = (r - rin) / (req - rin) 
   _x(r) = ( s = _s(r); 
             1 / (1 + a * s^pin / (1 + s^(pin - pcut))) )

   # now we want to normalize this to map to [-1, 1]
   # linear mapping to [-1, 1]
   #   x -> y(r) = 2 * (x - xcut) / (xin - xcut) - 1             
   #             = b0 * x + b1
   #  where b0 = 2 / (xin - xcut), b1 = -1 - 2 * xcut / (xin - xcut)
   xin = _x(rin)
   xcut = _x(rcut)
   b0 = 2 / (xin - xcut)
   b1 = -1 - 2 * xcut / (xin - xcut)

   # parameter structure / named-tuple 
   params = (pin = pin, pcut = pcut, 
             a = a, b0 = b0, b1 = b1, 
             rin = rin, req = req)
   
   return params 
end


""" 
   eval_gagnesi(r::Real, params)

Evaluate the generalized Agnesi transform at distance r, with parameters 
provided produced by `gagnesi_params`. 
"""
function eval_gagnesi(r::Real, params::NamedTuple) 
   pin, pcut, a, b0, b1, rin, req = 
         params.pin, params.pcut, params.a, params.b0, params.b1, 
         params.rin, params.req

   s = (r - rin) / (req - rin)
   x = 1 / (1 + a * s^(pin) / (1 + s^(pin - pcut)))
   y = b0 * x + b1
   return y 
end



@doc raw"""
   gagnesi_transform()

Construct a generalized Agnesi transform for  
   
   
```
trans = agnesi_transform(r0, p, q)
```
with `q >= p`. This generates an `AnalyticTransform` object that implements 
```math
   x(r) = \frac{1}{1 + a (r/r_0)^q / (1 + (r/r0)^(q-p))}
```
with default `a` chosen such that $|x'(r)|$ is maximised at $r = r_0$. But `a` may also be specified directly as a keyword argument. 

The transform satisfies 
```math 
   x(r) \sim \frac{1}{1 + a (r/r_0)^p} \quad \text{as} \quad r \to 0 
   \quad \text{and} 
   \quad 
   x(r) \sim \frac{1}{1 + a (r/r_0)^p}  \quad \text{as} r \to \infty.
```

As default parameters we recommend `p = 2, q = 4` and the defaults for `a`.
"""


function gagnesi_transform() 

end 
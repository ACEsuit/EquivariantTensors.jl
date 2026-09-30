# Small L = 0 coupling cases shared by test_trim_api.jl and test/trim/entry.jl.
const TRIM_NL = @NamedTuple{n::Int, l::Int}
const TRIM_LM = @NamedTuple{l::Int, m::Int}

# (maxn, maxl, max correlation order); (1, 2, 4) has degenerate nnll blocks,
# (1, 0, 8) reaches the table bound.
const TRIM_CASES = [(2, 2, 2), (2, 2, 3), (1, 2, 4), (1, 1, 5), (1, 0, 8)]

"""All sorted bodies of (n,l) channels (n ≤ maxn, l ≤ maxl) of length 1..ord with even ∑l."""
function trim_mb_spec(maxn::Int, maxl::Int, ord::Int)
   chans = [(n = n, l = l) for l in 0:maxl for n in 1:maxn]
   mb = Vector{TRIM_NL}[]
   bb = TRIM_NL[]
   function rec(start::Int)
      if !isempty(bb) && iseven(sum(b.l for b in bb))
         push!(mb, copy(bb))
      end
      length(bb) == ord && return
      for i in start:length(chans)
         push!(bb, chans[i]); rec(i); pop!(bb)
      end
   end
   rec(1)
   return mb
end

trim_rnl(maxn::Int, maxl::Int) = TRIM_NL[(n = n, l = l) for l in 0:maxl for n in 1:maxn]
trim_ylm(maxl::Int) = TRIM_LM[(l = l, m = m) for l in 0:maxl for m in -l:l]

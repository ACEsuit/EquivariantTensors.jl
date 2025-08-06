# ----------------------------------------------- 
# simple example embedding layer for radial and angular embeddings 
# to be replaced asap with a generic implementation. 
struct RnlYlmEmbedding{TTR, TBR, TTY, TBY}
   transr::TTR 
   rbasis::TBR 
   transy::TTY
   ybasis::TBY
end

initialparameters(rng::AbstractRNG, emb::RnlYlmEmbedding) = 
      (rbasis = NamedTuple(), ybasis = NamedTuple())

initialstates(rng::AbstractRNG, emb::RnlYlmEmbedding) = 
      (rbasis = NamedTuple(), ybasis = NamedTuple())

function evaluate(emb::RnlYlmEmbedding, X::ETGraph, ps, st)
   # Evaluate the radial and angular embeddings for the graph
   r = map(𝐫 -> emb.transr(𝐫), X.edge_data)
   Rnl = evaluate(emb.rbasis, r)
   R̂ = map(𝐫 -> 𝐫 / norm(𝐫), X.edge_data)
   Ylm = evaluate(emb.ybasis, R̂)

   # Reshape the embeddings into a 3D array format
   Rnl_3 = reshape_embedding(Rnl, X)
   Ylm_3 = reshape_embedding(Ylm, X)

   return (Rnl_3, Ylm_3), st 
end 


function ka_pullback(∂Rnl_3, ∂Ylm_3, emb::RnlYlmEmbedding, 
                     X::ETGraph, ps, st)
   # (re-)evaluate the radial and angular embeddings, now with 
   # derivatives to prepare for computing the pullback 
   r = map(𝐫 -> emb.transr(𝐫), X.edge_data)
   ∇r = map(𝐫 -> ForwardDiff.gradient(emb.transr, 𝐫), X.edge_data)
   Rnl, dRnl = evaluate_ed(emb.rbasis, r)

   R̂ = map(𝐫 -> 𝐫 / norm(𝐫), X.edge_data)
   Ylm, dYlm = evaluate_ed(emb.ybasis, R̂)

   ∂Rnl = rev_reshape_embedding(∂Rnl_3, X)
   ∂Ylm = rev_reshape_embedding(∂Ylm_3, X)

   # TODO: here I am assuming that edge_data is just a collection of 
   #       relative positions. This needs to be suitably generalized. 
   ∂𝐫_r = similar(X.edge_data)
   fill!(∂𝐫_r, zero(eltype(∂𝐫_r)))
   ∂𝐫_y = similar(X.edge_data)
   fill!(∂𝐫_y, zero(eltype(∂𝐫_y)))

   # ∇_𝐫[a] {∂Rnl : Rnl} 
   #   = ∇_𝐫[a] { ∑_a,k ∂Rnl[a,k] * Rnl[a,k] } 
   #   = ∑_k ∂Rnl[a,k] ∇_𝐫[a] Rnl[a,k]
   #   = { ∑_k ∂Rnl[a,k] dRnl[a,k] } * ∇𝐫[a]
   
   @kernel function _pb_Rnl!(∂𝐫, @Const(∂Rnl), @Const(dRnl), @Const(∇r))
      a = @index(Global)
      nfeats = size(dRnl, 2)
      t = ∂Rnl[a, 1] * dRnl[a, 1] 
      for k = 2:nfeats 
         t += ∂Rnl[a, k] * dRnl[a, k]
      end
      ∂𝐫[a] = t * ∇r[a] 
      nothing 
   end

   @kernel function _pb_Ylm!(∂𝐫, @Const(∂Ylm), @Const(dYlm), 
                                 @Const(R̂), @Const(r))
      a = @index(Global)
      nfeats = size(dYlm, 2)
      t = ∂Ylm[a, 1] * dYlm[a, 1]  # ℝ * ℝ³ → ℝ³
      for k = 2:nfeats
         t += ∂Ylm[a, k] * dYlm[a, k] 
      end 
      𝐫̂_a = R̂[a] 
      ∂𝐫[a] = t + 𝐫̂_a * (sum(𝐫̂_a .* t) / r[a])
      nothing 
   end

   nedg = nedges(X)
   backend = KernelAbstractions.get_backend(∂Rnl)
   
   kernel_pb_Rnl! = _pb_Rnl!(backend)
   kernel_pb_Rnl!(∂𝐫_r, ∂Rnl, dRnl, ∇r; ndrange = (nedg,))

   kernel_pb_Ylm! = _pb_Ylm!(backend)
   kernel_pb_Ylm!(∂𝐫_r, ∂Ylm, dYlm, R̂, r; ndrange = (nedg,))

   synchronize(backend)
   ∂𝐫 = ∂𝐫_r + ∂𝐫_y

   return ∂𝐫, st 
end


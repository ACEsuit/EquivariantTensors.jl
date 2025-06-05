
using MLDataDevices: AbstractDevice

struct PtClGraph{VECI, VECR}
   ii::VECI     # center particle indices / source indices
   jj::VECI     # neighbour particle indices / target indices
   first::VECI   # first[i] = first index of (i, j) pairs in ii, jj
   R::VECR      # relative positions / edge properties 
   nnodes::Int       # number of nodes in the graph
   maxneigs::Int     # maximum number of neighbors per node
end

(dev::AbstractDevice)(X::PtClGraph) = 
      PtClGraph(dev(X.ii), dev(X.jj), dev(X.first), dev(X.R), 
                X.nnodes, X.maxneigs)


struct RnYlmEmbedding{TTR, TBR, TTY, TBY}
   transr::TTR 
   rbasis::TBR 
   transy::TTY
   ybasis::TBY
end

function evaluate(emb::RnYlmEmbedding, graph::PtClGraph, ps, st)
   # Evaluate the radial and angular embeddings for the graph
   r = map(𝐫 -> emb.transr(𝐫), X.R)
   Rnl = evaluate(emb.rbasis, r, ps.rbasis, st.rbasis)
   R̂ = map(𝐫 -> 𝐫 / norm(𝐫), X.R)
   Ylm = evaluate(emb.ybasis, R̂, ps.ybasis, st.ybasis)

   # Reshape the embeddings into a 3D array format
   Rn_3 = reshape_embedding(Rn, X.ii, X.jj, X.nnodes, X.maxneigs)
   Ylm_3 = reshape_embedding(Ylm, X.ii, X.jj, X.nnodes, X.maxneigs)

   
end 



function reshape_embedding(P, ii, jj, nnodes, maxneigs)
   @kernel function _reshape_embedding!(P3, P, ii, jj, nnodes, maxneigs)
      a, ifeat = @index(Global, NTuple)
      i = ii[a]
      j = jj[a] 
      P3[j, i, ifeat] = P[a, ifeat]
      nothing 
   end
   
   nfeatures = size(P, 2)
   P3 = similar(P, (maxneigs, nnodes, nfeatures))
   fill!(P3, zero(eltype(P3)))
   backend = KernelAbstractions.get_backend(P3)
   kernel! = _reshape_embedding!(backend)
   kernel!(P3, P, ii, jj, nnodes, maxneigs; ndrange = size(P))
   return P3
end

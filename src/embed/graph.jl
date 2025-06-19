
using MLDataDevices: AbstractDevice
import LuxCore: AbstractLuxLayer, initialparameters, initialstates

struct ETGraph{VECI, TN, TE}
   ii::VECI     # center particle indices / source indices
   jj::VECI     # neighbour particle indices / target indices
   first::VECI   # first[i] = first index of (i, j) pairs in ii, jj
   node_data::TN     # node data 
   edge_data::TE     # edge data 
   maxneigs::Int     # maximum number of neighbors per node (for allocations)                      
end

nnodes(X::ETGraph) = length(X.first) - 1
nedges(X::ETGraph) = length(X.ii)
maxneigs(X::ETGraph) = X.maxneigs

function ETGraph(ii::AbstractVector{TI}, jj::AbstractVector{TI}; 
                 node_data = nothing, edge_data = nothing) where {TI} 
   if !issorted(ii) 
      error("i indices must be sorted")
   end

   nnodes = ii[end] 
   nedges = length(ii)

   # recompute the "first" array 
   first = similar(ii, (nnodes + 1,))
   first[1] = 1 
   idx = 1 
   for t = 1:length(ii) 
      if ii[t] > idx 
         while idx < ii[t]
            first[idx + 1] = t
            idx += 1
         end
      end
   end 
   first[end] = nedges + 1

   maxneigs = maximum(first[2:end] .- first[1:end-1])

   return ETGraph(ii, jj, first, node_data, edge_data, maxneigs)
end                 

(dev::AbstractDevice)(X::ETGraph) = 
      ETGraph(dev(X.ii), dev(X.jj), dev(X.first), 
              dev(X.node_data), dev(X.edge_data), X.maxneigs)

function neighbourhood(X::ETGraph, i::Int)
   # Returns the indices and edge data of the neighbours of node i
   #  (maybe it should also return node data of i and j ~ i??)
   first = X.first[i]
   last = X.first[i+1] - 1
   return X.jj[first:last], X.edge_data[first:last]
end


# ----------------------------------------------- 
# utility functions to work with the ETGraph and embedding it into 
# various formats for further processing

"""
   reshape_embedding(P, ii, jj, nnodes, maxneigs)

Takes a Nedges x Nfeat matrix and writes it into a 3-dimensional array of 
size (maxneigs, nnodes, Nfeat) where each column corresponds to a node. 
The "missing" neighbours are filled with zeros.
"""
function reshape_embedding(P, X::ETGraph)
   @kernel function _reshape_embedding!(P3, @Const(P), @Const(first))
      inode, ifeat = @index(Global, NTuple)
      i1 = first[inode]  
      i2 = first[inode + 1] - 1
      for t = 1:(i2-i1+1)
         iedge = i1 + t - 1  # edge index
         @inbounds P3[t, inode, ifeat] = P[iedge, ifeat]
      end
      nothing 
   end
   
   # size(P) == #edges x # features 
   nedges, nfeatures = size(P)
   P3 = similar(P, (maxneigs(X), nnodes(X), nfeatures))
   fill!(P3, zero(eltype(P3)))
   backend = KernelAbstractions.get_backend(P3)
   kernel! = _reshape_embedding!(backend)
   kernel!(P3, P, X.first; ndrange = (nnodes(X), nfeatures))
   KernelAbstractions.synchronize(backend)
   return P3
end

"""
   rev_reshape_embedding(P3, ii, jj, nnodes, maxneigs) -> P

Reverse operation for `reshape_embedding`. P3 is of shape 
(maxneigs, nnodes, nfeatures), and this gets written into P which is 
of shape (nedges, nfeatures) and then returned. 
"""
function rev_reshape_embedding(P3, X::ETGraph)
   @kernel function _rev_reshape_embedding!(P, @Const(P3), @Const(first))
      inode, ifeat = @index(Global, NTuple)
      i1 = first[inode]  
      i2 = first[inode + 1] - 1
      for t = 1:(i2-i1+1)
         iedge = i1 + t - 1  # edge index
         @inbounds P[iedge, ifeat] = P3[t, inode, ifeat]
      end
      nothing 
   end
   
   # size(P3) == maxneigs x #nodes x #features 
   nedg = nedges(X) 
   nfeatures = size(P3, 3) 
   P = similar(P3, (nedg, nfeatures))
   fill!(P, zero(eltype(P3)))

   backend = KernelAbstractions.get_backend(P)
   kernel! = _rev_reshape_embedding!(backend)
   kernel!(P, P3, X.first; ndrange = (nnodes(X), nfeatures))
   KernelAbstractions.synchronize(backend)
   return P3
end


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


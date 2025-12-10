
using MLDataDevices: AbstractDevice
import LuxCore: AbstractLuxLayer, initialparameters, initialstates

import Adapt
using Adapt: adapt 


struct ETGraph{VECI, TN, TE, TG}
   ii::VECI     # center particle indices / source indices
   jj::VECI     # neighbour particle indices / target indices
   first::VECI   # first[i] = first index of (i, j) pairs in ii, jj
   node_data::TN     # node data 
   edge_data::TE     # edge data 
   graph_data::TG    # graph data
   maxneigs::Int     # maximum number of neighbors per node (for allocations)                      
end

nnodes(X::ETGraph) = length(X.first) - 1
nedges(X::ETGraph) = length(X.ii)
maxneigs(X::ETGraph) = X.maxneigs

function ETGraph(ii::AbstractVector{TI}, jj::AbstractVector{TI}; 
                 node_data = nothing, edge_data = nothing, graph_data = nothing
                 ) where {TI}
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

   return ETGraph(ii, jj, first, node_data, edge_data, graph_data, maxneigs)
end               

function Adapt.adapt_structure(to, X::ETGraph) 
   ETGraph( adapt(to, X.ii), adapt(to, X.jj), adapt(to, X.first), 
            adapt(to, X.node_data), adapt(to, X.edge_data), 
            adapt(to, X.graph_data), X.maxneigs)
end

function (dev::AbstractDevice)(X::ETGraph) 
   ETGraph(dev(X.ii), dev(X.jj), dev(X.first), 
              dev(X.node_data), dev(X.edge_data), dev(X.graph_data), 
              X.maxneigs)
end

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
   return P
end



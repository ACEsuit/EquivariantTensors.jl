


module NeighbourListsExt

using NeighbourLists
import EquivariantTensors as ET
import NeighbourLists.AtomsBase: AbstractSystem

function ET.Atoms.interaction_graph(sys::AbstractSystem, rcut) 
   nlist = NeighbourLists.PairList(sys, rcut)
   return ET.Atoms.nlist2graph(nlist, sys)  
end

function ET.Atoms.nlist2graph(nlist::NeighbourLists.PairList, sys::AbstractSystem)
   ii = copy(nlist.i)
   jj = copy(nlist.j)
   first = copy(nlist.first) 
   R_ij = [ NeighbourLists._getR(nlist, n) for n = 1:length(ii) ] 
   S_i = [ NeighbourLists.AtomsBase.species(sys, i) for i in ii ] 
   S_j = [ NeighbourLists.AtomsBase.species(sys, j) for j in jj ]
   X_ij = [ (𝐫 = 𝐫, s0 = si, s1 = sj) for (𝐫, si, sj) in zip(R_ij, S_i, S_j) ]

   # for node data we use _only_ the atomic species for now so that we 
   # don't even give the option of using position information directly. 
   # ... until we sort out how to best handle this in ET. 
   X_i = [ (s = NeighbourLists.AtomsBase.species(sys, i),) 
           for i = 1:length(sys) ]

   G = ET.ETGraph(ii, jj; edge_data = X_ij, node_data = X_i)
   @assert G.first == first

   return G 
end 

function ET.Atoms.forces_from_edge_grads(sys::AbstractSystem, G::ET.ETGraph, ∇E_edges)
   # Given 𝐫_ij = X_j - X_i, and F = -∂E/∂X:
   # F[i] = -∂E/∂X_i = +∂E/∂𝐫_ij (since ∂𝐫_ij/∂X_i = -I)
   # F[j] = -∂E/∂X_j = -∂E/∂𝐫_ij (since ∂𝐫_ij/∂X_j = +I)

   TFRC = typeof(∇E_edges[1].𝐫)
   F = zeros(TFRC, length(sys))

   for (i, j, e) in zip(G.ii, G.jj, ∇E_edges)
      F[i] += e.𝐫
      F[j] -= e.𝐫
   end

   return F
end

function ET.Atoms.virial_from_edge_grads(G::ET.ETGraph, ∇E_edges)
   # Virial stress tensor: σ = -∑_edges (∂E/∂𝐫_ij) ⊗ 𝐫_ij
   # where 𝐫_ij is the edge position vector and ⊗ is outer product

   T = eltype(∇E_edges[1].𝐫)
   virial = zeros(T, 3, 3)

   for (edge_data, ∇E_edge) in zip(G.edge_data, ∇E_edges)
      𝐫ij = edge_data.𝐫
      ∂E_∂𝐫 = ∇E_edge.𝐫
      for α in 1:3, β in 1:3
         virial[α, β] -= ∂E_∂𝐫[α] * 𝐫ij[β]
      end
   end

   return virial
end

end

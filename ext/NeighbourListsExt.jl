


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

   G = ET.ETGraph(ii, jj; edge_data = X_ij)
   @assert G.first == first

   return G 
end 

function ET.Atoms.forces_from_edge_grads(sys::AbstractSystem, G::ET.ETGraph, ∇E_edges)
   
   TFRC = typeof(∇E_edges[1].𝐫)
   F = zeros(TFRC, length(sys)) 

   for (i, j, e) in zip(G.ii, G.jj, ∇E_edges) 
      F[i] -= e.𝐫
      F[j] += e.𝐫
   end

   return F
end

end

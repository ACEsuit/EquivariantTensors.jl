

module ETAtomsExt

using AtomsBase, NeighbourLists

import EquivariantTensors as ET
import Polynomials4ML as P4ML

function interaction_graph(sys::AbstractSystem, rcut) 
   nlist = NeighbourLists.PairList(sys, rcut)
   return nlist2graph(nlist, sys)  
end

function nlist2graph(nlist::NeighbourLists.PairList, sys::AbstractSystem)
   ii = copy(nlist.i)
   jj = copy(nlist.j)
   first = copy(nlist.first) 
   R_ij = [ NeighbourLists._getR(nlist, n) for n = 1:length(ii) ] 
   S_i = [ AtomsBase.species(sys, i) for i in ii ] 
   S_j = [ AtomsBase.species(sys, j) for j in jj ]
   X_ij = [ (𝐫 = 𝐫, s0 = si, s1 = sj) for (𝐫, si, sj) in zip(R_ij, S_i, S_j) ]

   G = ET.ETGraph(ii, jj; edge_data = X_ij)
   @assert G.first == first

   return G 
end 

end

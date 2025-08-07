#
# This example is incomplete since it doesn't yet allow the computation 
# of gradients i.e. forces. 
#
module MLIP

using AtomsBase, Lux, Random, NeighbourLists

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
   X_ij = [ (𝐫ij = 𝐫, si = si, sj = sj) for (𝐫, si, sj) in zip(R_ij, S_i, S_j) ]

   G = ET.ETGraph(ii, jj; edge_data = X_ij)
   @assert G.first == first

   return G 
end 

end

## 

using AtomsBase, Lux, Random, NeighbourLists, AtomsBuilder, Unitful 

import EquivariantTensors as ET
import Polynomials4ML as P4ML

##

sys = bulk(:Si, cubic=true) * (3,3,2) 
rcut = 5.0u"Å"
G_sys = MLIP.interaction_graph(sys, rcut)


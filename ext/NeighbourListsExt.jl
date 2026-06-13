


module NeighbourListsExt

using NeighbourLists
import EquivariantTensors as ET
# NeighbourLists 0.6 no longer re-exports AtomsBase/Unitful, so import
# AtomsBase directly (it re-exports `ustrip`). The `PairList(sys, rcut)`
# constructor is provided by NeighbourLists' own AtomsBase extension,
# which activates because AtomsBase is one of our triggers.
import AtomsBase
import AtomsBase: AbstractSystem
using DecoratedParticles: PState

function ET.Atoms.interaction_graph(sys::AbstractSystem, rcut) 
   nlist = NeighbourLists.PairList(sys, rcut)
   return ET.Atoms.nlist2graph(nlist, sys)  
end

function ET.Atoms.nlist2graph(nlist::NeighbourLists.PairList, sys::AbstractSystem)
   # NeighbourLists 0.6 returns Int32 index vectors; keep ETGraph indices
   # as `Int` (matches `Testing.rand_graph` and the maxneigs::Int field)
   ii = Int.(nlist.i)
   jj = Int.(nlist.j)
   first = copy(nlist.first)
   R_ij = [ NeighbourLists._getR(nlist, n) for n = 1:length(ii) ] 
   S_i = [ AtomsBase.species(sys, i) for i in ii ]
   S_j = [ AtomsBase.species(sys, j) for j in jj ]
   X_ij = [ PState(𝐫 = 𝐫, z0 = si, z1 = sj, 𝐒 = shift) 
            for (𝐫, si, sj, shift) in zip(R_ij, S_i, S_j, nlist.S) ]

   # for node data we use _only_ the atomic species for now so that we 
   # don't even give the option of using position information directly. 
   # ... until we sort out how to best handle this in ET. 
   X_i = [ PState(𝐫 = AtomsBase.ustrip.(AtomsBase.position(sys, i)),
                  z = AtomsBase.species(sys, i))
           for i = 1:length(sys) ]

   cell_vecs_u = AtomsBase.cell_vectors(sys)
   cell_vecs = ntuple( i -> AtomsBase.ustrip.(cell_vecs_u[i]),
                       length(cell_vecs_u) )

   sys_data = ( pbc = AtomsBase.periodicity(sys),
               cell = cell_vecs
              )

   G = ET.ETGraph(ii, jj; 
                  edge_data = X_ij, 
                  node_data = X_i, 
                  graph_data = sys_data)
   @assert G.first == first

   return G
end

end

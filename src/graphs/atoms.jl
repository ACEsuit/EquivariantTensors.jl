
# provide prototypes for extensions loaded via Pkg Extensions 

module Atoms

# Prototypes for NeighbourListsExt
# -----------------------------

"""
   interaction_graph(sys, rcut) -> ETGraph

Build an `ETGraph` from an AtomsBase system `sys` with cutoff radius
`rcut`, using a NeighbourLists pair list. Edge data are `PState`s
carrying the relative position `𝐫`, the species pair `z0, z1`, and the
cell shift `𝐒`; node data carry position and species.

The implementation lives in `NeighbourListsExt`, which activates only
when `NeighbourLists`, `AtomsBase`, and `DecoratedParticles` are all
loaded (NeighbourLists is the search engine, AtomsBase provides the
system and its accessors, DecoratedParticles provides the `PState` edge/
node data). A downstream package depending on those three gets the method
automatically once they load; in the REPL load all three explicitly,
e.g. `using EquivariantTensors, NeighbourLists, AtomsBase,
DecoratedParticles`.
"""
function interaction_graph end

"""
   nlist2graph(nlist, sys) -> ETGraph

Convert an existing NeighbourLists `PairList` `nlist` (built for the
AtomsBase system `sys`) into an `ETGraph`. Same extension and loading
requirements as [`interaction_graph`](@ref).
"""
function nlist2graph end




end 
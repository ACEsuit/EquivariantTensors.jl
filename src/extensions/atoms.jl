
# provide prototypes for extensions loaded via Pkg Extensions 

module Atoms 

# Prototypes for AtomsBaseExt 
# ----------------------------- 

function bond_len end
function agnesi_transform end

# Prototypes for NeighbourListExt
# -----------------------------

function interaction_graph end

function nlist2graph end

function forces_from_edge_grads end

# New API additions for multithreading/GPU support
function for_each_edge end

function interaction_graph_legacy end




end 
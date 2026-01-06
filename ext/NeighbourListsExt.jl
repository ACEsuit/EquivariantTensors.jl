

module NeighbourListsExt

using NeighbourLists
using AtomsBase
using AtomsBase: AbstractSystem, species, position, cell_vectors, periodicity
using Unitful: ustrip
import EquivariantTensors as ET
using DecoratedParticles: PState

# Import new API functions
using NeighbourLists: neighbour_list, build_cell_list, SortedCellList,
                      for_each_neighbour, default_backend

# ============================================================================
# LazyETGraph - memory-efficient graph for lazy iteration
# ============================================================================

"""
    LazyETGraph

A lazy graph representation that wraps a `SortedCellList` for memory-efficient
iteration over edges without materializing all pairs. Use `for_each_edge` to
iterate over edges of a node.
"""
struct LazyETGraph{TSys, TCL, TN, TG}
    clist::TCL          # SortedCellList
    sys::TSys           # Original AtomsBase system (for species lookup)
    node_data::TN       # Vector of node PStates
    graph_data::TG      # NamedTuple with pbc and cell
end

ET.nnodes(G::LazyETGraph) = length(G.sys)

"""
    for_each_edge(f, G::LazyETGraph, i)

Iterate over all edges of node `i` in a lazy graph, calling `f(j, edge_data)`
for each neighbor `j` with edge data as a PState.
"""
function ET.Atoms.for_each_edge(f, G::LazyETGraph, i::Integer)
    for_each_neighbour(G.clist, i) do j, R, S
        si = species(G.sys, i)
        sj = species(G.sys, j)
        edge = PState(𝐫 = R, z0 = si, z1 = sj, 𝐒 = S)
        f(j, edge)
    end
end

# ============================================================================
# interaction_graph - main entry point (updated for new API)
# ============================================================================

"""
    interaction_graph(sys::AbstractSystem, rcut; backend=nothing, lazy=false, int_type=Int32)

Convert an AtomsBase system to an ETGraph using a cutoff radius.

# Arguments
- `sys`: AtomsBase system
- `rcut`: Cutoff radius (with Unitful units)

# Keyword Arguments
- `backend`: KernelAbstractions backend. Default `nothing` uses CPU with automatic multithreading.
- `lazy`: If `true`, return a `LazyETGraph` for memory-efficient iteration instead of materializing all pairs.
- `int_type`: Integer type for indices (default: `Int32`)

# Returns
- `ETGraph` (default) or `LazyETGraph` (if `lazy=true`)
"""
function ET.Atoms.interaction_graph(sys::AbstractSystem, rcut;
                                    backend = nothing,
                                    lazy::Bool = false,
                                    int_type::Type = Int32)
    # Use default backend (CPU with multithreading) if not specified
    be = isnothing(backend) ? default_backend() : backend

    if lazy
        # Return lazy graph wrapping SortedCellList
        clist = build_cell_list(sys, rcut; backend=be, int_type=int_type)
        return _build_lazy_graph(clist, sys)
    else
        # Return materialized ETGraph (uses new multithreaded neighbour_list)
        nlist = neighbour_list(sys, rcut; backend=be, lazy=false, int_type=int_type)
        return ET.Atoms.nlist2graph(nlist, sys)
    end
end

"""
    _build_lazy_graph(clist::SortedCellList, sys::AbstractSystem)

Internal helper to construct a LazyETGraph from a SortedCellList.
"""
function _build_lazy_graph(clist::SortedCellList, sys::AbstractSystem)
    # Build node_data (same as materialized version)
    X_i = [ PState(𝐫 = ustrip.(position(sys, i)),
                   z = species(sys, i))
            for i = 1:length(sys) ]

    # Build graph_data
    cell_vecs_u = cell_vectors(sys)
    cell_vecs = ntuple(i -> ustrip.(cell_vecs_u[i]),
                       length(cell_vecs_u))
    sys_data = (pbc = periodicity(sys),
                cell = cell_vecs)

    return LazyETGraph(clist, sys, X_i, sys_data)
end

# ============================================================================
# interaction_graph_legacy - for comparison testing against old implementation
# ============================================================================

"""
    interaction_graph_legacy(sys::AbstractSystem, rcut)

Build an ETGraph using the legacy PairList constructor (linked-list algorithm).
This is retained for comparison testing against the new multithreaded implementation.
"""
function ET.Atoms.interaction_graph_legacy(sys::AbstractSystem, rcut)
    # Use the old PairList constructor (linked-list based)
    nlist = NeighbourLists.PairList(sys, rcut)
    return ET.Atoms.nlist2graph(nlist, sys)
end

# ============================================================================
# nlist2graph - convert PairList to ETGraph (unchanged from original)
# ============================================================================

function ET.Atoms.nlist2graph(nlist::NeighbourLists.PairList, sys::AbstractSystem)
   ii = copy(nlist.i)
   jj = copy(nlist.j)
   first = copy(nlist.first)
   R_ij = [ NeighbourLists._getR(nlist, n) for n = 1:length(ii) ]
   S_i = [ species(sys, i) for i in ii ]
   S_j = [ species(sys, j) for j in jj ]
   X_ij = [ PState(𝐫 = 𝐫, z0 = si, z1 = sj, 𝐒 = shift)
            for (𝐫, si, sj, shift) in zip(R_ij, S_i, S_j, nlist.S) ]

   # for node data we use _only_ the atomic species for now so that we
   # don't even give the option of using position information directly.
   # ... until we sort out how to best handle this in ET.
   X_i = [ PState(𝐫 = ustrip.(position(sys, i)),
                  z = species(sys, i))
           for i = 1:length(sys) ]

   cell_vecs_u = cell_vectors(sys)
   cell_vecs = ntuple( i -> ustrip.(cell_vecs_u[i]),
                       length(cell_vecs_u) )

   sys_data = ( pbc = periodicity(sys),
               cell = cell_vecs
              )

   G = ET.ETGraph(ii, jj;
                  edge_data = X_ij,
                  node_data = X_i,
                  graph_data = sys_data)
   @assert G.first == first

   return G
end

# ============================================================================
# forces_from_edge_grads - unchanged from original
# ============================================================================

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

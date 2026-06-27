
# using Pkg; Pkg.activate(@__DIR__() * "/../..")
# using TestEnv; TestEnv.activate() 

##

# need to load NeighbourLists to trigger the atoms extension
using EquivariantTensors, NeighbourLists, AtomsBase, AtomsBuilder, Unitful,
      Test, ACEbase, StaticArrays

using ACEbase.Testing: println_slim, print_tf
import EquivariantTensors as ET

##


@info("Convert a structure to an ETGraph + basic consistency tests")

sys = rattle!(bulk(:Si, cubic=true) * (3,3,2), 0.1u"Å")
rcut = 5.0u"Å"
G_sys = ET.Atoms.interaction_graph(sys, rcut)

nlist = NeighbourLists.PairList(sys, rcut)

println_slim(@test G_sys.graph_data.pbc == periodicity(sys)) 
println_slim(@test cell_vectors(sys) == (G_sys.graph_data.cell .* u"Å")) 
println_slim(@test [ x.𝐫 * u"Å" for x in G_sys.node_data ]  == position(sys, :)) 
println_slim(@test sort(nlist.i) == sort(G_sys.ii))
println_slim(@test sort(nlist.j) == sort(G_sys.jj))

##

@info("node_grads_from_edge_grads scatter")

# small hand-checkable graph: 3 nodes, explicit directed edges
ii = [1, 1, 2, 3]
jj = [2, 3, 3, 1]
G = ET.ETGraph(ii, jj)
w = [ SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0),
      SVector(0.0, 0.0, 1.0), SVector(1.0, 1.0, 1.0) ]

g = ET.node_grads_from_edge_grads(G, w)

# manual scatter: -w on the source node, +w on the target node
gman = [ zero(SVector{3, Float64}) for _ in 1:3 ]
for (i, j, we) in zip(ii, jj, w)
   gman[i] -= we
   gman[j] += we
end

println_slim(@test g == gman)
# every edge contributes ±w, so the total node gradient sums to zero
println_slim(@test sum(g) ≈ zero(SVector{3, Float64}))

## 


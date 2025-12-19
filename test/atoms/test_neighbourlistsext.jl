
# using Pkg; Pkg.activate(@__DIR__() * "/../..")
# using TestEnv; TestEnv.activate() 

##

using EquivariantTensors, NeighbourLists, AtomsBase, AtomsBuilder, Unitful, 
      Test, ACEbase

using ACEbase.Testing: println_slim, print_tf      

##


@info("Convert a structure to an ETGraph + basic consistency tests")

# need to load NeighbourLists to trigger the atoms extension 
using AtomsBuilder, NeighbourLists, Unitful 
import EquivariantTensors as ET

sys = rattle!(bulk(:Si, cubic=true) * (3,3,2), 0.1u"Å")
rcut = 5.0u"Å"
G_sys = ET.Atoms.interaction_graph(sys, rcut)

nlist = NeighbourLists.PairList(sys, rcut)

println_slim(@test G_sys.graph_data.pbc == periodicity(sys)) 
println_slim(@test (cell_vectors(sys) .* u"Å") == G_sys.graph_data.cell) 
println_slim(@test [ x.𝐫 * u"Å" for x in G_sys.node_data ]  == position(sys, :)) 
println_slim(@test sort(nlist.i) == sort(G_sys.ii))
println_slim(@test sort(nlist.j) == sort(G_sys.jj))

## 


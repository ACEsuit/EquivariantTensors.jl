
# using Pkg; Pkg.activate(@__DIR__() * "/../..")
# using TestEnv; TestEnv.activate()

##

using EquivariantTensors, NeighbourLists, AtomsBase, AtomsBuilder, Unitful,
      Test, ACEbase, LinearAlgebra

using ACEbase.Testing: println_slim, print_tf

##

@info("Test 1: Convert a structure to an ETGraph + basic consistency tests")

# need to load NeighbourLists to trigger the atoms extension
using AtomsBuilder, NeighbourLists, Unitful
import EquivariantTensors as ET

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

@info("Test 2: Validate new API vs legacy PairList implementation")

G_new = ET.Atoms.interaction_graph(sys, rcut)
G_legacy = ET.Atoms.interaction_graph_legacy(sys, rcut)

# Compare sorted indices (pairs may be in different order due to different algorithms)
println_slim(@test sort(G_new.ii) == sort(G_legacy.ii))
println_slim(@test sort(G_new.jj) == sort(G_legacy.jj))
println_slim(@test length(G_new.edge_data) == length(G_legacy.edge_data))

# Compare edge data by matching (i,j) pairs
function get_edge_dict(G)
    Dict((i,j) => e for (i,j,e) in zip(G.ii, G.jj, G.edge_data))
end

edges_new = get_edge_dict(G_new)
edges_legacy = get_edge_dict(G_legacy)

all_edges_match = true
for (k, v) in edges_legacy
    if !haskey(edges_new, k)
        all_edges_match = false
        break
    end
    if !isapprox(v.𝐫, edges_new[k].𝐫; atol=1e-10)
        all_edges_match = false
        break
    end
end
println_slim(@test all_edges_match)

##

@info("Test 3: Lazy mode produces same edges as materialized")

G_lazy = ET.Atoms.interaction_graph(sys, rcut; lazy=true)
G_mat = ET.Atoms.interaction_graph(sys, rcut)

# Count edges via lazy iteration and accumulate edge vectors
# Use Ref to avoid Julia scoping issue with do-blocks
edge_count = Ref(0)
edge_sum = Ref(zeros(3))
for i in 1:ET.nnodes(G_lazy)
    ET.Atoms.for_each_edge(G_lazy, i) do j, edge
        edge_count[] += 1
        edge_sum[] .+= edge.𝐫
    end
end

# Compare with materialized
println_slim(@test edge_count[] == length(G_mat.edge_data))
expected_sum = sum(e.𝐫 for e in G_mat.edge_data)
println_slim(@test isapprox(edge_sum[], expected_sum; rtol=1e-10))

# Verify lazy graph node data matches materialized
println_slim(@test length(G_lazy.node_data) == length(G_mat.node_data))
println_slim(@test G_lazy.graph_data.pbc == G_mat.graph_data.pbc)

##

@info("Test 4: Scaling tests with different system sizes")

for mult in [(2,2,2), (4,4,4), (5,5,5)]
    sys_scaled = rattle!(bulk(:Si, cubic=true) * mult, 0.1u"Å")
    G = ET.Atoms.interaction_graph(sys_scaled, rcut)
    println_slim(@test ET.nnodes(G) == length(sys_scaled))
    println_slim(@test length(G.ii) > 0)

    # Also test lazy mode works
    G_lazy = ET.Atoms.interaction_graph(sys_scaled, rcut; lazy=true)
    println_slim(@test ET.nnodes(G_lazy) == length(sys_scaled))
end

##

@info("Test 5: GPU transfer test (if GPU available)")

include(joinpath(@__DIR__(), "..", "test_utils", "utils_gpu.jl"))

if dev !== identity
    G_gpu = dev(G_sys)
    # Check that arrays are on GPU (CuArray or similar)
    println_slim(@test !(G_gpu.ii isa Vector))  # Not a CPU vector
    println_slim(@test length(G_gpu.ii) == length(G_sys.ii))
    println_slim(@test length(G_gpu.edge_data) == length(G_sys.edge_data))
else
    @info "No GPU available, skipping GPU transfer test"
end

##


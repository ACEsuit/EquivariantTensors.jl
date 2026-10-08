
# using Pkg; Pkg.activate(@__DIR__() * "/../..")
# using TestEnv; TestEnv.activate() 

##

# need to load NeighbourLists to trigger the atoms extension 
using EquivariantTensors, NeighbourLists, AtomsBase, AtomsBuilder, Unitful, 
      Test, ACEbase, LinearAlgebra

using ACEbase.Testing: println_slim, print_tf      
import EquivariantTensors as ET

##


@info("Test 1: Convert a structure to an ETGraph + basic consistency tests")

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

@info("Test 2: Validate new API vs legacy linked-list PairList implementation")

G_new = ET.Atoms.interaction_graph(sys, rcut)
G_legacy = ET.Atoms.interaction_graph_legacy(sys, rcut)

# Compare sorted indices (pairs may be in different order due to different algorithms)
println_slim(@test sort(G_new.ii) == sort(G_legacy.ii))
println_slim(@test sort(G_new.jj) == sort(G_legacy.jj))
println_slim(@test length(G_new.edge_data) == length(G_legacy.edge_data))

# Compare the displacement vectors edge by edge. Edges are grouped by (i, j)
# since a pair may have several periodic images within the cutoff; within a
# group the vectors are sorted so the two lists can be compared directly.
# The cell shift 𝐒 is deliberately NOT compared: the legacy code wraps
# positions into the cell (`fixcell=true`) before computing shifts, whereas
# the new implementation keeps 𝐒 relative to the positions as given.
function get_edge_vectors(G)
    D = Dict{Tuple{Int, Int}, Vector{typeof(G.edge_data[1].𝐫)}}()
    for (i, j, e) in zip(G.ii, G.jj, G.edge_data)
        push!(get!(D, (Int(i), Int(j)), eltype(valtype(D))[]), e.𝐫)
    end
    for v in values(D)
        sort!(v; by = 𝐫 -> (norm(𝐫), 𝐫[1], 𝐫[2], 𝐫[3]))
    end
    return D
end

edges_new = get_edge_vectors(G_new)
edges_legacy = get_edge_vectors(G_legacy)

println_slim(@test Set(keys(edges_new)) == Set(keys(edges_legacy)))
println_slim(@test all( length(edges_new[k]) == length(edges_legacy[k]) &&
                        all( isapprox(a, b; atol = 1e-10)
                             for (a, b) in zip(edges_new[k], edges_legacy[k]) )
                        for k in keys(edges_legacy) ))

# The new implementation's shifts must be consistent with the unwrapped
# positions: 𝐫_ij = X_j - X_i + cell' * 𝐒_ij
X_sys = [ ustrip.(u"Å", position(sys, i)) for i = 1:length(sys) ]
cell_mat = hcat(G_new.graph_data.cell...)     # columns are the lattice vectors
println_slim(@test all( isapprox(e.𝐫, X_sys[j] - X_sys[i] + cell_mat * e.𝐒; atol = 1e-10)
                        for (i, j, e) in zip(G_new.ii, G_new.jj, G_new.edge_data) ))

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
    local sys_scaled = rattle!(bulk(:Si, cubic=true) * mult, 0.1u"Å")
    local G = ET.Atoms.interaction_graph(sys_scaled, rcut)
    println_slim(@test ET.nnodes(G) == length(sys_scaled))
    println_slim(@test length(G.ii) > 0)

    # Also test lazy mode works
    local G_lazy_scaled = ET.Atoms.interaction_graph(sys_scaled, rcut; lazy=true)
    println_slim(@test ET.nnodes(G_lazy_scaled) == length(sys_scaled))
end

##

@info("Test 5: GPU transfer test (if GPU available)")

include(joinpath(@__DIR__(), "..", "test_utils", "utils_gpu.jl"))

if dev !== identity
    # Float32 so that the test also runs on Metal (no Float64 support)
    G_gpu = dev(ET.float32(G_sys))
    # Check that arrays are on GPU (CuArray or similar)
    println_slim(@test !(G_gpu.ii isa Vector))  # Not a CPU vector
    println_slim(@test length(G_gpu.ii) == length(G_sys.ii))
    println_slim(@test length(G_gpu.edge_data) == length(G_sys.edge_data))
else
    @info "No GPU available, skipping GPU transfer test"
end

##

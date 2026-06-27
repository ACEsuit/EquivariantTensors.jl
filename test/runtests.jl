using EquivariantTensors
using Test

include(joinpath(@__DIR__(), "test_utils", "utils_testO3.jl")) 
include(joinpath(@__DIR__(), "test_utils", "utils_gpu.jl"))

##

@testset "EquivariantTensors.jl" begin

@testset "Utils" begin
    @testset "SetProduct" begin include("utils/test_setproduct.jl"); end
    @testset "InvMap" begin include("utils/test_invmap.jl"); end
    @testset "Initializers" begin include("utils/test_initializers.jl"); end
    @testset "SelectLinL" begin include("utils/test_selectlinl.jl"); end
end

@testset "Embed" begin 
    @testset "Transform" begin include("embed/test_transform.jl"); end
    @testset "Decorated Particles" begin include("test_decoratedparticles.jl"); end
end

@testset "Pooling" begin
    @testset "StaticProd" begin include("utils/test_static_prod.jl"); end
    @testset "SparseProdPool" begin include("pooling/test_sparseprodpool.jl"); end
end

@testset "Sparse Format" begin
    @testset "SparseSymmetricProduct" begin include("formats/sparse/test_sparsesymmprod.jl"); end
    @testset "SparseMatrix-KA" begin include("formats/sparse/test_sparsemat_ka.jl"); end
end

@testset "Groups: O3" begin
    @testset "SYYVector" begin include("groups/O3/test_yyvector.jl"); end
    @testset "Clebsch Gordan Coeffs" begin include("groups/O3/test_clebschgordans.jl"); end
    @testset "Representation" begin include("groups/O3/test_representation.jl"); end
    @testset "Real AA to Complex AA" begin include("groups/O3/test_rAA2cAA.jl"); end
    @testset "Coupling Coeffs" begin include("groups/O3/test_coupling.jl"); end
    @testset "Coupling Coeffs with refl_sym given" begin include("groups/O3/test_coupling_augmented.jl"); end
    @testset "O3 Transformations" begin include("groups/O3/test_O3_transforms.jl"); end
    @testset "QuadO3" begin include("groups/O3/test_quad_O3.jl"); end
end

@testset "ACE Models" begin
    @testset "Pullback" begin include("acemodels/test_sparse_ace.jl"); end
    @testset "Pullback complex" begin include("acemodels/test_sparse_ace_cplx.jl"); end
    @testset "ACE prototype (ext)" begin include("acemodels/test_ace_ext.jl"); end
    @testset "End-to-end model" begin include("acemodels/test_model.jl"); end
end

@testset "Graphs" begin
    @testset "NeighbourListsExt" begin include("graphs/test_neighbourlistsext.jl"); end
end

end

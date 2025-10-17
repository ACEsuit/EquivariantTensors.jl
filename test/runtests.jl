using EquivariantTensors
using Test

include(joinpath(@__DIR__(), "test_utils", "utils_testO3.jl")) 
include(joinpath(@__DIR__(), "test_utils", "utils_gpu.jl"))

##

@testset "EquivariantTensors.jl" begin

@testset "Utils" begin 
    @testset "SetProduct" begin include("utils/test_setproduct.jl"); end
    @testset "InvMap" begin include("utils/test_invmap.jl"); end
end

@testset "Embed" begin 
    @testset "NamedTuples" begin include("embed/test_diffnt.jl"); end
    @testset "Transform" begin include("embed/test_transform.jl"); end
    @testset "ParallelEmbed" begin include("embed/test_parallelembed.jl"); end
end

@testset "ACE Layers" begin 
    @testset "StaticProd" begin include("ace/test_static_prod.jl"); end 
    @testset "SparseProdPool" begin include("ace/test_sparseprodpool.jl"); end 
    @testset "SparseSymmetricProduct" begin include("ace/test_sparsesymmprod.jl"); end 
    @testset "SparseMatrix-KA" begin include("ace/test_sparsemat_ka.jl"); end
end

@testset "O3-Coupling" begin 
    @testset "SYYVector" begin include("test_yyvector.jl"); end
    @testset "Clebsch Gordan Coeffs" begin include("test_clebschgordans.jl"); end
    @testset "Representation" begin include("test_representation.jl"); end
    @testset "Real AA to Complex AA" begin include("test_rAA2cAA.jl"); end
    @testset "Coupling Coeffs" begin include("test_coupling.jl"); end
end

@testset "ACE Models" begin 
    @testset "Pullback" begin include("acemodels/test_sparse_ace.jl"); end
    @testset "Pullback complex" begin include("acemodels/test_sparse_ace_cplx.jl"); end
    @testset "ACE KA and grad" begin include("acemodels/test_ace_ka.jl"); end
    @testset "ACE KA new version" begin include("acemodels/test_ace_ka2.jl"); end
end 

end

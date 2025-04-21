using EquivariantTensors
using Test

isdefined(Main, :___UTILS_FOR_TESTS___) || include("utils/utils_testO3.jl")

##

@testset "EquivariantTensors.jl" begin

    @testset "ACE" begin 
        @testset "SparseProdPool" begin
            include("ace/test_static_prod.jl")
            include("ace/test_sparseprodpool.jl")
            include("ace/test_sparsesymmprod.jl")
        end
        @testset "SparseSymmetricProduct" begin
            include("ace/test_sparsesymmprod.jl")
        end
    end

    @testset "O3-Coupling" begin 
        @testset "SYYVector" begin include("test_yyvector.jl"); end
        @testset "Clebsch Gordan Coeffs" begin include("test_clebschgordans.jl"); end
        @testset "Representation" begin include("test_representation.jl"); end
        @testset "Coupling Coeffs" begin include("test_coupling.jl"); end
    end 
end

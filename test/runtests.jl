using EquivariantTensors
using Test

isdefined(Main, :___UTILS_FOR_TESTS___) || include("utils/utils_testO3.jl")

##

@testset "EquivariantTensors.jl" begin

    @testset "Utils" begin 
        @testset "SetProduct" begin include("utils/test_setproduct.jl"); end
    end

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
        @testset "Recursive Coupling Coeffs" begin include("test_recursive_coupling.jl"); end
    end 
    # @testset "O3 new" begin include("new_rpe_test.jl"); end
        
end

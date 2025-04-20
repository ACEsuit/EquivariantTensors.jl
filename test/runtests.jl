using EquivariantTensors
using Test

isdefined(Main, :___UTILS_FOR_TESTS___) || include("utils/utils_testO3.jl")

##

@testset "EquivariantTensors.jl" begin

    @testset "SparseProdPool" begin
        include("ace/test_sparseprodpool.jl")
    end

    @testset "O3-Coupling" begin 
        @testset "SYYVector" begin include("test_yyvector.jl"); end
        @testset "Clebsch Gordan Coeffs" begin include("test_clebschgordans.jl"); end
        @testset "Representation" begin include("test_representation.jl"); end
        @testset "Coupling Coeffs" begin include("test_coupling.jl"); end
    end 
end

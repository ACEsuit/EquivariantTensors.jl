using ACEradials
using Test

@testset "ACEradials.jl" begin
    @testset "Learnable & Splined Rnl" begin include("test_radials.jl"); end
    @testset "Agnesi transform" begin include("test_agnesi.jl"); end
end

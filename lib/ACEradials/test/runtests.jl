using ACEradials
using Test

@testset "ACEradials.jl" begin
    @testset "Learnable & Splined Rnl" begin include("test_radials.jl"); end
end

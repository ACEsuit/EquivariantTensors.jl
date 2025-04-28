
using Test, EquivariantTensors
using ACEbase.Testing: println_slim

##


@info("Testing the setproduct function")

function _test_product(A)
   P1 = EquivariantTensors.setproduct(A)
   pp = Iterators.product(A...)   
   P2 = reduce(vcat, [ transpose([p...]) for p in pp ])
   sort(P1, dims = 1) == sort(P2, dims = 1)
end

A = [ -2:2, -1:1, 0:0 ]
println_slim( @test _test_product(A) )

A = [ -2:2, -1:1, 0:0, 1:3 ]
println_slim( @test _test_product(A) )

A = [ [1,2,3], [4,5,6], [7.7,8,9] ]
println_slim( @test _test_product(A) )

A = [ [1,true,3], [4,5,6], [7.7,8,9] ]
println_slim( @test _test_product(A) )

# for now we allow failure on emptyproducts, this is a bit tricky to get 
# right without type instability ... 
# Aempty = [ [1,2,3], [] ]
# EquivariantTensors.setproduct(Aempty)

##
# profiling code in case we need it again 

# using BenchmarkTools
# @btime EquivariantTensors.setproduct( $([ (-5:5), (-2:2), (-3:3) ]) )

# @code_warntype EquivariantTensors.setproduct( ([ (-5:5), (-2:2), (-3:3) ]) )

# @profview let A = [ (-5:5), (-2:2), (-3:3) ]
#    for _ = 1:1_000_000 
#       P = EquivariantTensors.setproduct(A)
#    end
# end

using EquivariantTensors, Test 
import EquivariantTensors as ET 

##

function rand_b() 
   l = rand(1:12) 
   return (n = rand(1:20), l = rand(1:10), m = rand(-l:l) )
end

rand_bb() = sort( [ rand_b() for _ = 1:rand(1:4) ] )

##

@info("Testing invmap")

N = 100_000 
A = unique( [ rand_bb() for _ = 1:N ] )

# @time 
invA = ET.invmap(A)

println(@test(
   all( invA[a] == i for (i, a) in enumerate(A) )
   ))

B = [ A[rand(1:10)] for _ = 1:100 ]
try 
   invB = ET.invmap(B)
   @test false 
catch e 
   println(@test typeof(e) == ArgumentError)
end
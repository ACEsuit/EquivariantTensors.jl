using Test, EquivariantTensors, ChainRulesCore, Random
using EquivariantTensors: PooledSparseProduct, evaluate, evaluate!, 
         _generate_input, _generate_input_1
using ACEbase.Testing: fdtest, println_slim, print_tf 
ET = EquivariantTensors

function _generate_basis(; order=3, len = 50)
    NN = [ rand(10:30) for _ = 1:order ]
    spec = sort([ ntuple(t -> rand(1:NN[t]), order) for _ = 1:len])
    return PooledSparseProduct(spec)
end

function test_ka_evaluate(basis = order -> _generate_basis(; order=order); 
                          generate_x = basis -> _generate_input(basis), 
                          nXrg = 32:100, 
                          ntest = 8, 
                          dev = Array)
    @info("Testing KA implementation of $basis")
    for _ = 1:ntest       
        order = mod1(ntest, 4)
        basis = _generate_basis(; order=order)                         
        bBB = generate_x(basis) 
        nX = size(bBB[1], 1)
        P1 = evaluate(basis, bBB)
        evaluate!(P1, basis, bBB)
        P2 = dev(similar(P1)) 
        ET.ka_evaluate!(P2, basis, bBB, nX)
        print_tf(@test P1 ≈ P2)
    end
    println() 
    return nothing 
end

test_ka_evaluate()

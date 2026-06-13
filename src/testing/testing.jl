module Testing 

import EquivariantTensors as ET
using StaticArrays, Random, LinearAlgebra

function rand_ball(rcut::T)  where {T}
   u = randn(SVector{3, T})
   r = (0.001 + rand() * rcut) / (0.001 + rcut)
   return r * (u / norm(u))
end

# returns PState(𝐫 = rand_ball(rcut)); the method is provided by
# ext/DecoratedParticlesExt.jl since the default edge data needs the
# state arithmetic from DecoratedParticles (e.g. in pullbacks)
function _default_randedge end

function rand_graph(nnodes;
                    nneigrg = min(nnodes ÷ 2 + 1, 10):min(nnodes, 30),
                    T = Float32,
                    rcut = one(T),
                    randedge = () -> _default_randedge(rcut), )
   ii = Int[] 
   jj = Int[]
   rmax = nnodes^(1/3) * 0.5
   maxneigs = 0 
   edges = []
   for i in 1:nnodes
      nneig = rand(nneigrg)
      neigs_i = shuffle(1:nnodes)[1:nneig] 
      for t in 1:nneig
         push!(ii, i)
         push!(jj, neigs_i[t])
         push!(edges, randedge())
      end
   end
   return ET.ETGraph(ii, jj; edge_data = identity.(edges)) 
end

end

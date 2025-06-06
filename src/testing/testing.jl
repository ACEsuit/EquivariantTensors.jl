module Testing 

import EquivariantTensors as ET
using StaticArrays, Random, LinearAlgebra

function rand_graph(nnodes;
                    nneigrg = 20:40,  
                    T = Float32, 
                    rcut = one(T))
   ii = Int[] 
   jj = Int[]
   Rij = SVector{3, T}[]
   rmax = nnodes^(1/3) * 0.5
   maxneigs = 0 
   for i in 1:nnodes
      nneig = rand(nneigrg)
      neigs_i = shuffle(1:nnodes)[1:nneig] 
      for t in 1:nneig
         push!(ii, i)
         push!(jj, neigs_i[t])
         u = randn(SVector{3, T})
         r = (0.001 + rand() * rcut) / (0.001 + rmax) 
         push!(Rij, r * u / norm(u))
      end
   end
   return ET.ETGraph(ii, jj; edge_data = Rij) 
end

end
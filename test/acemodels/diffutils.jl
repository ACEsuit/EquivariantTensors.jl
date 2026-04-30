
module DIFF 

using Zygote, ForwardDiff, Optimisers, StaticArrays
import EquivariantTensors as ET
import DecoratedParticles as DP

grad_zy(X, model, ps, st) = Zygote.gradient(G -> model(G, ps, st)[1], X)[1]

grad_zy_ps(X, model, ps, st) = Zygote.gradient(_ps -> model(X, _ps, st)[1], ps)[1]

function grad_fd(G, model, ps, st) 

   function replace_edges(X, Rmat)
      Rsvec = [ SVector{3}(Rmat[:, i]) for i in 1:size(Rmat, 2) ]
      new_edgedata = [ (; 𝐫 = 𝐫) for 𝐫 in Rsvec ]
      return ET.ETGraph( X.ii, X.jj, X.first, 
                  X.node_data, new_edgedata, X.graph_data, 
                  X.maxneigs )
   end 

   function _eval_mat(Rmat)
      G_new = replace_edges(G, Rmat)
      return model(G_new, ps, st)[1]
   end
      
   Rsvec = [ x.𝐫 for x in G.edge_data ]
   Rmat = reinterpret(reshape, eltype(Rsvec[1]), Rsvec)
   ∇E_fd = ForwardDiff.gradient(_eval_mat, Rmat)
   ∇E_svec = [ SVector{3}(∇E_fd[:, i]) for i in 1:size(∇E_fd, 2) ]
   ∇E_edges = [ DP.VState(; 𝐫 = 𝐫) for 𝐫 in ∇E_svec ]
   return ET.ETGraph( G.ii, G.jj, G.first, 
               G.node_data, ∇E_edges, G.graph_data, 
               G.maxneigs )
end 

function grad_fd_ps(G, model, ps, st)
   p_flat, rebuild = destructure(ps)
   _eval_p(p) = model(G, rebuild(p), st)[1]
   ∇p_flat = ForwardDiff.gradient(_eval_p, p_flat)
   return rebuild(∇p_flat)
end

end
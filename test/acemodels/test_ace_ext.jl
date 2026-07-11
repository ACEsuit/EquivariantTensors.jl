
# Extended ACE tests built on the hand-coded `ACEKA` prototype — a `SimpleACE`
# with a manual KA basis evaluation, a semi-analytic `evaluate_with_grad`, and a
# basis Jacobian. This complements `test_model.jl` (which covers the
# `SparseACElayer` model + CPU/GPU consistency); here we exercise the prototype's
# gradients and Jacobian on CPU. Graph-structure tests live in `test/graphs/`.

using LinearAlgebra, Lux, Random, EquivariantTensors, Test, StaticArrays,
      Zygote, ForwardDiff, DecoratedParticles
using ACEbase.Testing: print_tf, println_slim
import EquivariantTensors as ET
import Polynomials4ML as P4ML
import SpheriCart
import Optimisers: destructure

##

module ACEKA

   using LinearAlgebra, Random, Zygote
   using DecoratedParticles: VState
   import LuxCore: initialparameters, initialstates
   import ChainRulesCore: rrule

   import EquivariantTensors as ET
   import KernelAbstractions as KA

   struct SimpleACE{T, TR, TY, BB}
      Rnl::TR
      Ylm::TY
      symbasis::BB    # symmetric basis
      params::Vector{T}   # model parameters
   end

   initialparameters(rng::AbstractRNG, m::SimpleACE) =
            (      Rnl = initialparameters(rng, m.Rnl),
                   Ylm = initialparameters(rng, m.Ylm),
              symbasis = initialparameters(rng, m.symbasis),
                params = copy(m.params), )

   initialstates(rng::AbstractRNG, m::SimpleACE) =
            (      Rnl = initialstates(rng, m.Rnl),
                   Ylm = initialstates(rng, m.Ylm),
              symbasis = initialstates(rng, m.symbasis), )

   # 𝔹 = (#nodes, #features); params = (#features, #readouts)
   # in this toy model, #readouts = 1.

   function eval_basis(model::SimpleACE, X::ET.ETGraph, ps, st)
      Rnl, _ = model.Rnl(X, ps.Rnl, st.Rnl)
      Ylm, _ = model.Ylm(X, ps.Ylm, st.Ylm)
      (𝔹,), _ = ET.ka_evaluate(model.symbasis, Rnl, Ylm, ps.symbasis, st.symbasis)
      return 𝔹
   end

   function evaluate(model::SimpleACE, X::ET.ETGraph, ps, st)
      𝔹 = eval_basis(model, X, ps, st)
      return 𝔹 * ps.params, st
   end

   function jacobian_basis(model::SimpleACE, X::ET.ETGraph, ps, st)
      (R, ∂R), _ = ET.evaluate_ed(model.Rnl, X, ps.Rnl, st.Rnl)
      (Y, ∂Y), _ = ET.evaluate_ed(model.Ylm, X, ps.Ylm, st.Ylm)
      𝔹, ∂𝔹 = ET._jacobian_X(model.symbasis, R, Y, ∂R, ∂Y,
                              ps.symbasis, st.symbasis)
      return 𝔹, ∂𝔹
   end

   # Semi-manual gradients are still much more efficient
   #
   function evaluate_with_grad(model::SimpleACE, X::ET.ETGraph, ps, st)
      (Rnl, dRnl), _ = ET.evaluate_ed(model.Rnl, X, ps.Rnl, st.Rnl)
      (Ylm, dYlm), _ = ET.evaluate_ed(model.Ylm, X, ps.Ylm, st.Ylm)

      (𝔹,), A, 𝔸 = ET._ka_evaluate(model.symbasis, Rnl, Ylm,
                  st.symbasis.aspec, st.symbasis.aaspecs, st.symbasis.A2Bmaps)
      φ = 𝔹 * ps.params

      # let's assume we eventually produce E = ∑φ then ∂E = 1, which
      # backpropagates to ∂φ = (1,1,1...)
      # ∂E/∂𝔹 = ∂/∂𝔹 { 1ᵀ 𝔹 params } = ∂/∂𝔹 { 𝔹 : 1 ⊗ params}
      ∂𝔹 = fill!(similar(𝔹, (size(𝔹, 1),)), one(eltype(𝔹))) * ps.params'

      # packpropagate through the symmetric basis
      ∂Rnl, ∂Ylm = ET._ka_pullback((∂𝔹,), model.symbasis, Rnl, Ylm, A, 𝔸,
                                    st.symbasis.aspec, st.symbasis.aaspecs, st.symbasis.A2Bmaps)

      # this could be made more memory efficient by avoiding the
      # many intermediate allocations
      _grad_R = ET._pullback_edge_embedding(∂Rnl, dRnl, X)
      _grad_Y = ET._pullback_edge_embedding(∂Ylm, dYlm, X)

      return φ, _grad_R .+ _grad_Y
   end

end

##
# ---- build the prototype model + a random input graph (CPU, Float32) ----

Dtot = 12   # total degree (truncation of embeddings & correlations)
maxl = 8    # maximum degree of spherical/solid harmonics
ORD  = 3    # correlation order (body-order = ORD + 1)

rbasis = P4ML.ChebBasis(Dtot+1)
rembed = ET.EdgeEmbed(ET.StateEmbed(ET.state_transform(x -> 1 / (1 + norm(x.𝐫)^2)), rbasis))
ybasis = SpheriCart.SolidHarmonics(maxl; static=true)
yembed = ET.EdgeEmbed(ET.StateEmbed(ET.state_transform(x -> x.𝐫), ybasis))

mb_spec = ET.sparse_nnll_set(; L = 0, ORD = ORD, minn = 0, maxn = Dtot, maxl = maxl,
                  level = bb -> sum((b.n + b.l) for b in bb; init=0), maxlevel = Dtot)
𝔹basis = ET.sparse_equivariant_tensor(; L = 0, mb_spec = mb_spec,
            Rnl_spec = P4ML.natural_indices(rbasis),
            Ylm_spec = P4ML.natural_indices(ybasis), basis = real)
θ = randn(Float32, length(𝔹basis, 0))

model = ACEKA.SimpleACE(rembed, yembed, 𝔹basis, θ)
ps, st = LuxCore.setup(MersenneTwister(1234), model)
ps = ET.float32(ps); st = ET.float32(st)

X = ET.float32(ET.Testing.rand_graph(30; nneigrg = 5:10))

# scalar energy of a graph G at parameters p
energy(G, p) = sum(ACEKA.evaluate(model, G, p, st)[1])

##
# ============ `evaluate`: gradient w.r.t. positions X ============
# Zygote, ForwardDiff (over the edge positions), and the hand-coded
# `evaluate_with_grad` must all agree.

@info("evaluate: ∂/∂X  (Zygote vs ForwardDiff vs manual)")

∇X_zy = Zygote.gradient(G -> energy(G, ps), X)[1]

function _grad_X_fd(G)
   _replace(Rmat) = ET.ETGraph(G.ii, G.jj, G.first, G.node_data,
         [ PState(𝐫 = SVector{3}(Rmat[:, i])) for i in 1:size(Rmat, 2) ],
         G.graph_data, G.maxneigs)
   Rmat = reinterpret(reshape, eltype(G.edge_data[1].𝐫), [ x.𝐫 for x in G.edge_data ])
   ∇ = ForwardDiff.gradient(R -> energy(_replace(R), ps), Rmat)
   return [ SVector{3}(∇[:, i]) for i in 1:size(∇, 2) ]
end
∇X_fd = _grad_X_fd(X)

_, ∂X_man = ACEKA.evaluate_with_grad(model, X, ps, st)

∇X_zy_𝐫  = [ x.𝐫 for x in ∇X_zy.edge_data ]
∇X_man_𝐫 = [ x.𝐫 for x in ∂X_man ]
println_slim(@test ∇X_fd ≈ ∇X_zy_𝐫 ≈ ∇X_man_𝐫)

##
# ============ `evaluate`: gradient w.r.t. parameters ============

@info("evaluate: ∂/∂params  (Zygote vs ForwardDiff)")

gp_zy = Zygote.gradient(p -> energy(X, p), ps)[1].params
gp_fd = ForwardDiff.gradient(θ -> energy(X, merge(ps, (; params = θ))), ps.params)
println_slim(@test gp_zy ≈ gp_fd)

##
# ============ `evaluate_with_grad`: gradient w.r.t. parameters ============
# A made-up scalar loss of `evaluate_with_grad`'s outputs (energies φ and the
# position gradient ∂X), differentiated w.r.t. the parameters. Differentiating
# *through* the hand-written pullback is not yet supported (no rrule), so the
# Zygote path is currently broken. TODO (agents/tests.md): make
# `evaluate_with_grad` differentiable w.r.t. parameters.

@info("evaluate_with_grad: ∂/∂params  (currently broken — TODO)")

function ewg_loss(p)
   φ, ∂X = ACEKA.evaluate_with_grad(model, X, p, st)
   return sum(abs2, φ) + sum(sum(abs2, dx.𝐫) for dx in ∂X)
end
# Both paths currently fail: `evaluate_with_grad` allocates Float32 buffers and
# uses a hand-written pullback with no rrule, so neither ForwardDiff (Dual) nor
# Zygote can differentiate it w.r.t. the parameters. Wrapped in @test_broken so
# the error is recorded (not thrown).
@test_broken ForwardDiff.gradient(θ -> ewg_loss(merge(ps, (; params = θ))), ps.params) ≈
             Zygote.gradient(ewg_loss, ps)[1].params

##
# ============ Jacobian of the basis w.r.t. positions ============
# Check it matches the (contracted) Zygote gradient and the basis itself.

@info("jacobian_basis vs contracted gradient")

𝔹3, ∂𝔹3 = ACEKA.jacobian_basis(model, X, ps, st)
∂𝔹2 = ET.rev_reshape_embedding(∂𝔹3[1], X)
∂𝔹2xθ = ∂𝔹2 * θ
println_slim(@test 𝔹3[1] ≈ ACEKA.eval_basis(model, X, ps, st))
println_slim(@test all(∇X_zy.edge_data .≈ ∂𝔹2xθ))

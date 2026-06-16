#
# CP / TRACE equivariant tensor format — correctness tests.
#
# Three independent checks (agents/trace.md §6 testing plan):
#   1. Sparse oracle    — for a single radial channel with rank K=1 and W=1 the
#                         CP basis is bit-identical to the established sparse
#                         B-basis (the reference format).
#   2. Equivariance     — F_0 is rotation-invariant; F_1 is covariant with the
#                         real Wigner-D extracted from the SH basis.
#   3. FD gradients     — Zygote vs finite differences w.r.t. the input features
#                         A and the parameters (W, λ).
#

using EquivariantTensors, StaticArrays, Random, LinearAlgebra, Test
using LuxCore, Zygote
using ACEbase.Testing: println_slim, print_tf
import EquivariantTensors as ET
import Polynomials4ML as P4ML

rng = MersenneTwister(1234)

##

@info("CP/TRACE: sparse oracle (K=1, W=1 ≡ sparse B-basis)")

let
   maxl = 2
   ybasis = P4ML.real_sphericalharmonics(maxl)
   yidx = P4ML.natural_indices(ybasis)
   Rnl_spec = [(n = 1,)]
   mb1 = [ [(n=1,l=0)], [(n=1,l=1)],
           [(n=1,l=0),(n=1,l=0)], [(n=1,l=1),(n=1,l=1)],
           [(n=1,l=0),(n=1,l=1)],
           [(n=1,l=0),(n=1,l=0),(n=1,l=0)],
           [(n=1,l=1),(n=1,l=1),(n=1,l=0)] ]

   sparse = ET.sparse_equivariant_tensors(; LL=(0,1), mb_spec=mb1,
               Rnl_spec=Rnl_spec, Ylm_spec=yidx, basis=real)
   cp = ET.cp_equivariant_tensor(; LL=(0,1), mb_spec=mb1,
               Rnl_spec=Rnl_spec, Ylm_spec=yidx, basis=real, rank=1)

   println_slim(@test length(cp, 0) == length(sparse, 0))
   println_slim(@test length(cp, 1) == length(sparse, 1))

   A = randn(rng, 1, length(cp.abasis.spec))
   _, stc = LuxCore.setup(rng, cp)
   psc = (W = [ ones(1, 1) for _ = 1:length(cp.mixer.nl_count) ],)
   BBcp, _ = cp(A, psc, stc)

   AAs = ET.evaluate(sparse.aabasis, A[1, :])
   Bsp = [ sparse.A2Bmaps[iL] * AAs for iL = 1:2 ]
   println_slim(@test BBcp[1][1, 1, :] ≈ Bsp[1])
   println_slim(@test BBcp[2][1, 1, :] ≈ Bsp[2])
end

##

@info("CP/TRACE: equivariance (L=0 invariant, L=1 covariant)")

let
   Dtot = 5; maxl = 3; ORD = 3; K = 4
   rbasis = P4ML.legendre_basis(Dtot+1)
   ybasis = P4ML.real_sphericalharmonics(maxl)
   yidx = P4ML.natural_indices(ybasis)
   mb = ET.sparse_nnll_set(; ORD=ORD, minn=0, maxn=Dtot, maxl=maxl,
           level = bb -> sum((b.n+b.l) for b in bb; init=0), maxlevel=Dtot)
   basis = ET.cp_equivariant_tensor(; LL=(0,1), mb_spec=mb,
           Rnl_spec=P4ML.natural_indices(rbasis), Ylm_spec=yidx, basis=real, rank=K)
   layer = ET.CPACElayer(basis, (1, 1))
   ps, st = LuxCore.setup(rng, layer)

   rand_rot() = (Q = @SMatrix randn(3, 3); exp(Q - Q'))
   rand_ball() = (u = randn(rng, SVector{3, Float64}); u / norm(u) * rand(rng))

   poolA(Rs) = reshape(
         ET.evaluate(basis.abasis,
            (reduce(vcat, [ P4ML.evaluate(rbasis, norm(r))' for r in Rs ]),
             reduce(vcat, [ P4ML.evaluate(ybasis, r)'       for r in Rs ]))),
         1, :)

   # real Wigner-D for block L extracted from the SH basis: Y_L(Qr) = D Y_L(r)
   function wignerD(L, Q)
      colsL = findall(b -> b.l == L, yidx)
      pts = [ rand_ball() for _ = 1:(4*(2L+1)) ]
      Y  = reduce(hcat, [ P4ML.evaluate(ybasis, p)[colsL]   for p in pts ])
      YQ = reduce(hcat, [ P4ML.evaluate(ybasis, Q*p)[colsL] for p in pts ])
      return YQ / Y
   end

   for _ = 1:8
      Rs = [ rand_ball() for _ = 1:6 ]
      Q = rand_rot()
      D1 = wignerD(1, Q)
      o,  _ = layer(poolA(Rs), ps, st)
      oQ, _ = layer(poolA([ Q*r for r in Rs ]), ps, st)
      print_tf(@test abs(o[1][1, 1] - oQ[1][1, 1]) < 1e-9)          # L=0
      print_tf(@test norm(oQ[2][1, 1] - D1 * o[2][1, 1]) < 1e-8)    # L=1
   end
   println()
end

##

@info("CP/TRACE: EquivLinearL primitive (mix n, identity on m) + gradient")

let
   Dtot = 4; maxl = 2; ORD = 2; K = 3
   rbasis = P4ML.legendre_basis(Dtot+1)
   ybasis = P4ML.real_sphericalharmonics(maxl)
   mb = ET.sparse_nnll_set(; ORD=ORD, minn=0, maxn=Dtot, maxl=maxl,
           level = bb -> sum((b.n+b.l) for b in bb; init=0), maxlevel=Dtot)
   basis = ET.cp_equivariant_tensor(; LL=(0,), mb_spec=mb,
           Rnl_spec=P4ML.natural_indices(rbasis),
           Ylm_spec=P4ML.natural_indices(ybasis), basis=real, rank=K)
   mixer = basis.mixer
   ps, st = LuxCore.setup(rng, mixer)
   A = randn(rng, 5, length(basis.abasis.spec))

   Ā, _ = mixer(A, ps, st)
   println_slim(@test size(Ā) == (5, K, mixer.len))

   mloss(A, W) = sum(abs2, ET._eql_apply(mixer, A, W))
   gA = Zygote.gradient(a -> mloss(a, ps.W), A)[1]
   gA_fd = zero(A); h = 1e-6
   for i in eachindex(A)
      Ap = copy(A); Ap[i]+=h; Am = copy(A); Am[i]-=h
      gA_fd[i] = (mloss(Ap, ps.W) - mloss(Am, ps.W)) / (2h)
   end
   println_slim(@test gA ≈ gA_fd)
end

##

@info("CP/TRACE: finite-difference gradients (A, W, λ)")

let
   Dtot = 5; maxl = 2; ORD = 2; K = 3
   rbasis = P4ML.legendre_basis(Dtot+1)
   ybasis = P4ML.real_sphericalharmonics(maxl)
   mb = ET.sparse_nnll_set(; ORD=ORD, minn=0, maxn=Dtot, maxl=maxl,
           level = bb -> sum((b.n+b.l) for b in bb; init=0), maxlevel=Dtot)
   basis = ET.cp_equivariant_tensor(; LL=(0,1), mb_spec=mb,
           Rnl_spec=P4ML.natural_indices(rbasis),
           Ylm_spec=P4ML.natural_indices(ybasis), basis=real, rank=K)
   layer = ET.CPACElayer(basis, (2, 2))
   ps, st = LuxCore.setup(rng, layer)
   A = randn(rng, 4, length(basis.abasis.spec))

   loss(A, ps) = ( o = layer(A, ps, st)[1];
                   sum(abs2, o[1]) + sum(sum(abs2, x) for x in o[2]) )
   h = 1e-6

   # ∂/∂A
   gA_zy = Zygote.gradient(a -> loss(a, ps), A)[1]
   gA_fd = zero(A)
   for i in eachindex(A)
      Ap = copy(A); Ap[i] += h; Am = copy(A); Am[i] -= h
      gA_fd[i] = (loss(Ap, ps) - loss(Am, ps)) / (2h)
   end
   println_slim(@test gA_zy ≈ gA_fd)

   # ∂/∂(W, λ)
   gps = Zygote.gradient(p -> loss(A, p), ps)[1]
   W = ps.basis.W
   okW = true
   for il in eachindex(W), idx in eachindex(W[il])
      Wp = deepcopy(W); Wp[il][idx] += h
      Wm = deepcopy(W); Wm[il][idx] -= h
      g = (loss(A, (basis=(W=Wp,), λ=ps.λ)) -
           loss(A, (basis=(W=Wm,), λ=ps.λ))) / (2h)
      okW &= isapprox(gps.basis.W[il][idx], g; atol=1e-5, rtol=1e-4)
   end
   println_slim(@test okW)

   λ = ps.λ
   okλ = true
   for iL in eachindex(λ), idx in eachindex(λ[iL])
      λp = deepcopy(λ); λp[iL][idx] += h
      λm = deepcopy(λ); λm[iL][idx] -= h
      g = (loss(A, (basis=ps.basis, λ=Tuple(λp))) -
           loss(A, (basis=ps.basis, λ=Tuple(λm)))) / (2h)
      okλ &= isapprox(gps.λ[iL][idx], g; atol=1e-5, rtol=1e-4)
   end
   println_slim(@test okλ)
end

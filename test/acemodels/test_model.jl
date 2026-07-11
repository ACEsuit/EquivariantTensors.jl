
# End-to-end ACE model tests: `embed -> SparseACElayer -> readout`. This is the
# only place `SparseACElayer` (its `et_normal` init, `evaluate`, and the
# `_tupmul` / `_mul_scal` rrules) and the parameter gradients through it are
# exercised. CPU checks always run (native Float64); GPU consistency runs only
# when a real backend is detected, over the supported precisions. See
# agents/tests.md. The body is wrapped in a `let` to avoid polluting `Main`
# (these files are all `include`d into `Main`).

using LinearAlgebra, Lux, Random, Test
using ACEbase.Testing: print_tf, println_slim
using EquivariantTensors
import EquivariantTensors as ET

include(joinpath(@__DIR__(), "..", "test_utils", "utils_gpu.jl"))
include(joinpath(@__DIR__(), "..", "test_utils", "luxtestmodels.jl"))
include(joinpath(@__DIR__(), "..", "test_utils", "diffutils.jl"))

let Dtot = 8, maxl = 6, ORD = 3

   build(LL, NFEAT, readout) = begin
      embed, acel = LTM.build_model(; Dtot, maxl, ORD, LL, NFEAT)
      Lux.Chain(; embed = embed, ace = acel, readout = WrappedFunction(readout))
   end

   # (name, model): invariant / equivariant / mixed output features
   models = [
      ("L0, 2 feat (invariant)",
         build((0,), (2,), U -> sum(U[1][:,1]) + 0.1f0 * sum(U[1][:,2].^2)) ),
      ("L1, 1 feat (equivariant)",
         build((1,), (1,), U -> sum(abs2, sum(U[1]))) ),
      ("L0+L1, 3+1 feat (mixed)",
         build((0,1), (3,1),
            U -> (U0 = U[1]; U1 = U[2];
                  sum(U0[:,1]) + 0.1f0*sum(U0[:,2].^2) + 0.01f0*sum(U0[:,3].^3)
                  + sum(abs2, sum(U1))) ) ),
   ]

   X = ET.Testing.rand_graph(30; nneigrg = 5:10)

   # ---- CPU checks (always; native Float64 for a well-conditioned guardrail):
   #      forward + Zygote-vs-ForwardDiff on positions and parameters. The
   #      parameter check exercises SparseACElayer's WLL + rrules.
   cpu_checks(model) = begin
      ps, st = LuxCore.setup(MersenneTwister(1234), model)
      φ, _ = model(X, ps, st)
      g_zy = DIFF.grad_zy(X, model, ps, st)
      g_fd = DIFF.grad_fd(X, model, ps, st)
      g_ps    = DIFF.grad_zy_ps(X, model, ps, st)
      g_ps_fd = DIFF.grad_fd_ps(X, model, ps, st)
      (; isnum = φ isa Number,
         pos_ok = all(g_zy.edge_data .≈ g_fd.edge_data),
         par_ok = all(g_ps.ace.WLL .≈ g_ps_fd.ace.WLL))
   end

   @info("End-to-end model tests (CPU)")
   for (name, model) in models
      r = cpu_checks(model)
      print_tf(@test r.isnum)
      print_tf(@test r.pos_ok)
      print_tf(@test r.par_ok)
   end
   println()

   # ---- GPU consistency (only when a real backend is detected) ----
   # F32 always; F64 only where the backend supports it (Metal is F32-only).
   # Move host<->device with `dev` (recursive, handles ps/st and ETGraph).
   if gpu_backend != "CPU"
      precisions = gpu_supports_f64 ? [(Float64, identity), (Float32, ET.float32)] :
                                      [(Float32, ET.float32)]
      for (T, cv) in precisions, (name, model) in models
         @info("GPU consistency [$(T)]: $(name)")
         ps, st = LuxCore.setup(MersenneTwister(1234), model)
         ps = cv(ps); st = cv(st); Xc = cv(X)
         ps_d = dev(ps); st_d = dev(st); X_d = dev(Xc)
         rtol = T == Float32 ? 1.0f-3 : 1e-6

         # forward pass: CPU vs GPU
         φ   = model(Xc, ps, st)[1]
         φ_d = model(X_d, ps_d, st_d)[1]
         println_slim(@test isapprox(φ_d, φ; rtol = rtol))

         # parameter gradient: CPU vs GPU
         gW   = DIFF.grad_zy_ps(Xc, model, ps, st).ace.WLL
         gW_d = DIFF.grad_zy_ps(X_d, model, ps_d, st_d).ace.WLL
         println_slim(@test all(isapprox(Array(gW_d[i]), gW[i]; rtol = rtol)
                                for i in 1:length(gW)))

         # position gradient: CPU vs GPU. Correct on CUDA (verified on an A100,
         # F32 + F64); a known failure on Metal only (the original report). So a
         # real @test everywhere except @test_broken on Metal. See agents/tests.md.
         gz   = DIFF.grad_zy(Xc, model, ps, st)
         gz_d = DIFF.grad_zy(X_d, model, ps_d, st_d)
         if gpu_backend == "Metal"
            @test_broken all(Array(gz_d.edge_data) .≈ gz.edge_data)
         else
            println_slim(@test all(Array(gz_d.edge_data) .≈ gz.edge_data))
         end
      end
      println()
   end

end

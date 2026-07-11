using MLDataDevices
import Pkg

if !isdefined(Main, :___EQT_UTILS_GPU___)

   """
       detect_gpu_backend() -> String

   Pick a GPU backend by probing the *system* — no GPU package is loaded here, so
   the default CI runner resolves to `"CPU"` and installs no GPU package. Set the
   `TEST_BACKEND` env var to force a choice (`"CPU"`, `"CUDA"`, `"AMDGPU"`,
   `"Metal"`, `"oneAPI"`).
   """
   function detect_gpu_backend()
      haskey(ENV, "TEST_BACKEND") && return ENV["TEST_BACKEND"]   # manual override
      if Sys.isapple() && Sys.ARCH == :aarch64
         return "Metal"
      elseif !isnothing(Sys.which("nvidia-smi")) && success(`nvidia-smi`)
         return "CUDA"
      elseif !isnothing(Sys.which("rocm-smi")) || isdir("/dev/kfd")
         return "AMDGPU"
      elseif !isnothing(Sys.which("sycl-ls"))   # crude oneAPI probe
         return "oneAPI"
      else
         return "CPU"
      end
   end

   # When a GPU is detected, install the matching backend *into the (sandboxed)
   # test env* and use it; the default CI runner resolves to "CPU" and installs
   # nothing. A detected-but-unusable backend degrades to CPU with a warning so
   # the suite still runs.
   # Move objects host<->device with the `dev`/`gpu` *function* (recursive,
   # Adapt/Functors-aware — handles ps/st NamedTuples and ETGraph), not a bare
   # array type. `gpu_supports_f64` is false on F32-only backends (Metal); tests
   # run F32 always and F64 only when this is true.
   global gpu_backend = detect_gpu_backend()
   global gpu = global dev = identity
   global gpu_supports_f64 = true

   if gpu_backend != "CPU"
      try
         Pkg.add(gpu_backend)                  # into the sandboxed test env only
         @eval using $(Symbol(gpu_backend))
         if gpu_backend == "CUDA"
            @assert CUDA.functional();   global gpu = global dev = CUDA.cu
         elseif gpu_backend == "Metal"
            @assert Metal.functional();  global gpu = global dev = Metal.mtl
            global gpu_supports_f64 = false              # Metal is F32-only
         elseif gpu_backend == "AMDGPU"
            @assert AMDGPU.functional(); global gpu = global dev = MLDataDevices.gpu_device()
         elseif gpu_backend == "oneAPI"
            @assert oneAPI.functional(); global gpu = global dev = oneAPI.oneArray
         else
            error("unknown TEST_BACKEND = $(gpu_backend)")
         end
         @info "GPU test backend: $(gpu_backend) (F64 supported: $(gpu_supports_f64))"
      catch e
         @warn "GPU backend '$(gpu_backend)' detected but not usable; using CPU." exception=(e, catch_backtrace())
         global gpu_backend = "CPU"
         global gpu = global dev = identity
         global gpu_supports_f64 = true
      end
   end

   gpu_backend == "CPU" && @info "GPU test backend: CPU (dev = identity)."

   global ___EQT_UTILS_GPU___ = true

end

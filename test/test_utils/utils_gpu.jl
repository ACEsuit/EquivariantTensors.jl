
if !isdefined(Main, :___EQT_UTILS_GPU___)

   global __has_cuda = false    
   global __has_roc = false 
   global __has_metal = false

   try    
      using CUDA
      if CUDA.functional()
         @info "Using CUDA"
         CUDA.versioninfo()
         global gpu = CuArray
         global dev = CuArray
         global __has_cuda = true
      else 
         @info("CUDA is not functional")
      end
   catch e
      @info "Couldn't load CUDA"
   end

   if !__has_cuda
      try
         using AMDGPU
         if AMDGPU.functional()
            @info "Using AMD"
            AMDGPU.versioninfo()
            global gpu = ROCArray
            global dev = ROCArray
            global __has_roc = true
         else
            @info("AMDGPU is not functional")
         end
      catch e
         @info "Couldn't load AMDGPU"
      end
   end 

   if !__has_roc
      try
         using Metal
         if Metal.functional()
            @info "Using Metal"
            Metal.versioninfo()
            global gpu = Metal.mtl
            global dev = Metal.mtl
            global __has_metal = true
         else
            @info("Metal is not functional")
         end
      catch e
         @info "Couldn't load Metal"
      end
   end 

   if !__has_cuda && !__has_roc && !__has_metal
      @info "No GPU is available. Using CPU."
      global gpu = identity 
      global dev = identity 
   end

   global ___EQT_UTILS_GPU___ = true

end

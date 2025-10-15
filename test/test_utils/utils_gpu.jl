using MLDataDevices

if !isdefined(Main, :___EQT_UTILS_GPU___)

   global __has_cuda = false    
   global __has_roc = false 
   global __has_metal = false
   global gpu = global dev = nothing 

   try    
      using CUDA
      if CUDA.functional()
         @info "Using CUDA"
         CUDA.versioninfo()
         global __has_cuda = true
         global gpu = global dev = cu 
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
            global __has_roc = true
            global gpu = global dev = gpu_device() 
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
            global __has_metal = true
            global gpu = global dev = mtl 
         else
            @info("Metal is not functional")
         end
      catch e
         @info "Couldn't load Metal"
      end
   end 

   if !__has_cuda && !__has_roc && !__has_metal
      @info "No GPU is available. Using CPU."
   end

   

   global ___EQT_UTILS_GPU___ = true

end

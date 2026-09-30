# Compile test/trim/entry.jl with `juliac --trim=safe` and check the executable
# reproduces the uncompiled results.  Needs Julia >= 1.13:
#     julia +1.13 --project=test/trim test/trim/build_and_run.jl
VERSION >= v"1.13" || error("trim test needs Julia >= 1.13 (got $VERSION)")
const HERE = @__DIR__
const JC_ENV = joinpath(HERE, "juliac_env")
const JULIA = Base.julia_cmd()
run(`$JULIA --project=$HERE -e "using Pkg; Pkg.instantiate()"`)
run(`$JULIA --project=$JC_ENV -e "using Pkg; Pkg.instantiate()"`)
out = mktempdir()
run(Cmd(`$JULIA --project=$JC_ENV -e "using JuliaC; JuliaC.main(ARGS)" -- --output-exe trim_entry --project=$HERE --trim=safe --experimental --bundle $out $(joinpath(HERE, "entry.jl"))`; dir = out))
exe = joinpath(out, "bin", Sys.iswindows() ? "trim_entry.exe" : "trim_entry")
isfile(exe) || error("no executable at $exe; bundle has $(readdir(out))")
got = readlines(`$exe`)
ref = Core.eval(Main, :(module TrimEntryRef end))      # a real module: entry.jl calls include
Base.include(ref, joinpath(HERE, "entry.jl"))
want = Base.invokelatest(getfield(ref, :checksum_lines))
got == want || error("trimmed executable differs from Julia:\n got  = $got\n want = $want")
println("trim test OK: $(length(got)) cases")

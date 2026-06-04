using Pkg
using Test
using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

# Centralized sublibrary CI (SciML/.github sublibrary-tests.yml@v1) runs this
# root test suite with GROUP set to a sublibrary's CI group: the bare sublibrary
# name selects that sublibrary's "Core" group, and "<sublibrary>_<grp>" selects a
# named group. Detect those here, activate the sublibrary's own test environment,
# and hand off to its runtests.jl (which parses GROUP itself). GROUP="All" (the
# local default) and GROUP="Corleone" run the main-package suite below.
const LIB_DIR = joinpath(@__DIR__, "..", "lib")

function _detect_sublibrary(group, lib_dir)
    isdir(joinpath(lib_dir, group)) && return group
    for i in length(group):-1:1
        if group[i] == '_' && isdir(joinpath(lib_dir, group[1:(i - 1)]))
            return group[1:(i - 1)]
        end
    end
    return nothing
end

const _SUBLIB = _detect_sublibrary(GROUP, LIB_DIR)

if _SUBLIB !== nothing
    sublib_path = joinpath(LIB_DIR, _SUBLIB)
    Pkg.activate(sublib_path)
    # On Julia < 1.11 the [sources] table in Project.toml is ignored, so develop
    # the local path dependencies (e.g. Corleone) to test the PR branch code.
    if VERSION < v"1.11.0-DEV.0"
        toml = Pkg.TOML.parsefile(joinpath(sublib_path, "Project.toml"))
        if haskey(toml, "sources")
            specs = Pkg.PackageSpec[]
            for (dep, spec) in toml["sources"]
                if spec isa Dict && haskey(spec, "path")
                    dep_path = normpath(joinpath(sublib_path, spec["path"]))
                    isdir(dep_path) && push!(specs, Pkg.PackageSpec(path = dep_path))
                end
            end
            isempty(specs) || Pkg.develop(specs)
        end
    end
    # Forward GROUP unchanged; the sublibrary runtests.jl parses it.
    withenv("GROUP" => GROUP) do
        Pkg.test(_SUBLIB; allow_reresolve = true)
    end
else
    using JET
    @testset "Corleone.jl" begin
        @safetestset "Code quality (Aqua.jl)" begin
            using Aqua
            using Corleone
            Aqua.test_all(Corleone)
        end
        @safetestset "Local controls" begin
            include("local_controls.jl")
        end
        @safetestset "Multiple shooting" begin
            include("multiple_shooting.jl")
        end
        @testset "Examples" begin
            @safetestset "Lotka" begin
                include("examples/lotka_oc.jl")
            end
            @safetestset "Lotka MS" begin
                include("examples/lotka_ms.jl")
            end
            @safetestset "Lotka MTK" begin
                include("examples/mtk.jl")
            end
        end
    end
end

# What to test?
# local_controls.jl:
#   - construction of index_grid, get_subvector_indices -> Julius
# general:
#   - more convergence? Lotka OED

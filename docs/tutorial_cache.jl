module TutorialCache

using SHA: sha256
using TOML

export tutorial_context, tutorial_key, valid_tutorial_cache
export save_tutorial_cache!, restore_tutorial_cache!

const CACHE_VERSION = 1

file_digest(path) = bytes2hex(open(sha256, path))

function input_files(path)
    isfile(path) && return [abspath(path)]
    isdir(path) || throw(ArgumentError("Missing tutorial cache input: $path"))
    files = String[]
    for (directory, subdirectories, filenames) in walkdir(path)
        filter!(name -> name ∉ (".git", ".cache"), subdirectories)
        for name in filenames
            name in ("Manifest.toml", ".DS_Store") && continue
            any(suffix -> endswith(name, suffix), (".cov", ".mem")) && continue
            push!(files, abspath(joinpath(directory, name)))
        end
    end
    return sort!(files)
end

function fingerprint(files, root; context = "")
    io = IOBuffer()
    println(io, context)
    for path in sort!(unique(files))
        # Length-prefixed names avoid ambiguous concatenations. Contents, rather
        # than mtimes, also work after checkout or archive restoration.
        name = replace(relpath(path, root), '\\' => '/')
        println(io, ncodeunits(name), ":", name, ":", file_digest(path))
    end
    return bytes2hex(sha256(take!(io)))
end

function manifest_digest(root)
    path = joinpath(root, "docs", "Manifest.toml")
    isfile(path) || error("Instantiate the documentation environment before caching tutorials.")
    manifest = TOML.parsefile(path)
    # Pkg.develop records checkout paths. The code of these packages is hashed
    # separately, so relocating the checkout must not invalidate every entry.
    for name in ("Corleone", "CorleoneOED", "CorleoneInverse")
        for package in get(get(manifest, "deps", Dict()), name, [])
            haskey(package, "path") && (package["path"] = "checkout:$name")
        end
    end
    io = IOBuffer()
    TOML.print(io, manifest; sorted = true)
    return bytes2hex(sha256(take!(io)))
end

"Hash shared code, renderer, resolved dependencies, and execution platform once per build."
function tutorial_context(root)
    files = [
        joinpath(root, "Project.toml"), joinpath(root, "docs", "Project.toml"),
        joinpath(root, "docs", "process_tutorials.jl"), joinpath(root, "docs", "tutorial_cache.jl"),
    ]
    packages = [root]
    library_root = joinpath(root, "lib")
    isdir(library_root) && append!(packages, filter(isdir, readdir(library_root; join = true)))
    for package in packages
        project = joinpath(package, "Project.toml")
        isfile(project) && push!(files, project)
        for directory in ("src", "ext")
            path = joinpath(package, directory)
            isdir(path) && append!(files, input_files(path))
        end
    end
    runtime = join(
        (
            CACHE_VERSION, VERSION, Sys.KERNEL, Sys.ARCH, Sys.WORD_SIZE,
            Threads.nthreads(), manifest_digest(root),
        ), ":"
    )
    return fingerprint(files, root; context = runtime)
end

"Hash one example directory and explicitly declared cross-directory helper/data dependencies."
function tutorial_key(source, context; dependencies = String[], root = dirname(abspath(source)))
    dependencies isa AbstractVector && all(path -> path isa AbstractString, dependencies) ||
        throw(ArgumentError("cache_dependencies must be a list of paths relative to the example directory."))
    directory = dirname(abspath(source))
    files = input_files(directory)
    for dependency in dependencies
        append!(files, input_files(joinpath(directory, dependency)))
    end
    return fingerprint(files, root; context = context)
end

function entry_path(cache, name)
    occursin(r"^[A-Za-z0-9_-]+$", name) || throw(ArgumentError("Invalid tutorial cache name: $name"))
    return joinpath(cache, name)
end

function artifact_name(file, name)
    basename(file) == file || return false
    file in ("$name.md", "$name.jl") && return true
    startswith(file, "$name-") || return false
    return occursin(r"^-(?:figure-)?[0-9]+\.[A-Za-z0-9]+$", file[(ncodeunits(name) + 1):end])
end

"Validate complete output and all content digests before suppressing execution."
function valid_tutorial_cache(cache, name, key)
    entry = entry_path(cache, name)
    marker = joinpath(entry, "entry.toml")
    isfile(marker) || return false
    try
        metadata = TOML.parsefile(marker)
        get(metadata, "version", nothing) == CACHE_VERSION || return false
        get(metadata, "key", nothing) == key || return false
        files = metadata["files"]
        haskey(files, "$name.md") && haskey(files, "$name.jl") || return false
        for (file, digest) in files
            artifact_name(file, name) || return false
            path = joinpath(entry, file)
            isfile(path) && !islink(path) || return false
            file_digest(path) == digest || return false
        end
        return true
    catch exception
        exception isa InterruptException && rethrow()
        @warn "Ignoring unreadable tutorial cache" tutorial = name exception
        return false
    end
end

function save_tutorial_cache!(cache, name, key, output)
    destination = entry_path(cache, name)
    mkpath(cache)
    temporary = mktempdir(cache; prefix = ".writing-")
    try
        files = Dict{String, String}()
        for file in readdir(output)
            artifact_name(file, name) || continue
            path = joinpath(output, file)
            isfile(path) && !islink(path) || continue
            cp(path, joinpath(temporary, file))
            files[file] = file_digest(path)
        end
        haskey(files, "$name.md") && haskey(files, "$name.jl") ||
            error("Incomplete tutorial outputs for $name; refusing to cache them.")
        open(joinpath(temporary, "entry.toml"), "w") do io
            TOML.print(io, Dict("version" => CACHE_VERSION, "key" => key, "files" => files); sorted = true)
        end
        # Publish only after the full entry has been written. Keep just the latest
        # successful result for each tutorial to bound CI archive size.
        mv(temporary, destination; force = true)
    finally
        isdir(temporary) && rm(temporary; recursive = true)
    end
    return destination
end

function restore_tutorial_cache!(cache, name, key, output)
    valid_tutorial_cache(cache, name, key) || return false
    entry = entry_path(cache, name)
    files = TOML.parsefile(joinpath(entry, "entry.toml"))["files"]
    mkpath(output)
    for file in keys(files)
        cp(joinpath(entry, file), joinpath(output, file); force = true)
    end
    return true
end

end

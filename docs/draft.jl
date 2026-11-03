# GitHub checks out a synthetic merge commit for pull requests. Its first
# parent is the PR base, so this comparison covers the whole PR.
function documentation_draft(; root = joinpath(@__DIR__, ".."), env = ENV)
    get(env, "GITHUB_ACTIONS", "false") == "true" || return false
    ref = get(env, "GITHUB_REF", "")
    (ref == "refs/heads/main" || startswith(ref, "refs/tags/")) && return false
    get(env, "GITHUB_EVENT_NAME", "") == "pull_request" || return false

    try
        sha = env["GITHUB_SHA"]
        base = "$sha^1"
        # The reusable workflow uses checkout's default shallow clone.
        if !success(pipeline(`git -C $root cat-file -e $base`; stderr = devnull))
            run(`git -C $root fetch --no-tags --depth=2 origin $sha`)
        end
        tutorial_paths = [
            "examples/",
            "lib/CorleoneOED/examples/",
            "lib/OptimalControlBenchmarks/docs/src/index.md",
            "docs/process_tutorials.jl",
        ]
        result = run(ignorestatus(`git -C $root diff --quiet $base $sha -- $tutorial_paths`))
        result.exitcode == 0 && return true
        result.exitcode == 1 && return false
        error("git diff exited with status $(result.exitcode)")
    catch exception
        @warn "Could not check tutorial changes; running a full documentation build" exception
        return false
    end
end

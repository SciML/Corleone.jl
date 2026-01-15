module CorleoneOEDZygoteExtension

using CorleoneOED
using Zygote

@info "Loading CorleoneOEDZygoteExtension..."

function CorleoneOED._get_sampling_sums!(
    res::Zygote.Buffer, oed::OEDLayer{false,true,true}, x, ps, st, ::Val{RESET}
) where {RESET}
    (; active_controls, tspans) = st
    (; controls) = ps
    dts = CorleoneOED._get_dts(tspans)
    foreach(enumerate(active_controls)) do (i, subset)
        res[i] = sum(controls[subset] .* dts)
    end
end

function CorleoneOED._get_sampling_sums!(
    res::Zygote.Buffer, oed::OEDLayer{true,true}, x, ps, st, ::Val{RESET}
) where {RESET}
    (; sampling_indices) = oed
    (; active_controls) = st
    (; controls) = ps
    foreach(
        enumerate(CorleoneOED.__get_subsets(active_controls, sampling_indices))
    ) do (i, subset)
        if RESET
            res[i] = sum(controls[subset])
        else
            res[i] += sum(controls[subset])
        end
    end
end

function CorleoneOED._get_sampling_sums!(
    res::Zygote.Buffer, oed::OEDLayer{false,true,false}, x, ps, st, ::Val{RESET}
) where {RESET}
    (; sampling_indices,) = oed
    (; index_grid, tspans) = st
    (; controls) = ps
    dts = CorleoneOED._get_dts(tspans)
    foreach(
        enumerate(eachrow(CorleoneOED.__get_subsets(index_grid, sampling_indices)))
    ) do (i, subset)
        if RESET
            res[i] = sum(controls[subset] .* dts)
        else
            res[i] += sum(controls[subset] .* dts)
        end
    end
end

end
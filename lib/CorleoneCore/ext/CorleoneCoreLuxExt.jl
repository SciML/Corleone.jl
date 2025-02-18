module CorleoneCoreLuxExt

using CorleoneCore: CorleoneCore
using Lux: Lux 

CorleoneCore.InternalWrapper.is_extension_loaded(::Val{:Lux}) = true 

function wrap_stateful(d::LuxCore.AbstractLuxLayer, ps, st::NamedTuple, args...)
    Lux.StatefulLuxLayer{true}(d, ps, st)
end 

end
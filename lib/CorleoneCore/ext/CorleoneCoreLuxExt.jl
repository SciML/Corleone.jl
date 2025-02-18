module CorleoneCoreLuxExt

using CorleoneCore: CorleoneCore, LuxCore
using Lux: Lux 

@info "Loaded Extension"

CorleoneCore.InternalWrapper.is_extension_loaded(::Val{:Lux}) = true 

function CorleoneCore.InternalWrapper.wrap_stateful(d::LuxCore.AbstractLuxLayer, ps, st::NamedTuple, args...)
    Lux.StatefulLuxLayer{true}(d, ps, st)
end 

end
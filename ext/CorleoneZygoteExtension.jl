module CorleoneZygoteExtension

import Corleone:is_extension_loaded, unwrapbuffer

is_extension_loaded(::Val{:Zygote}) = true

unwrapbuffer(x::Zygote.Buffer) = copy(x)

end
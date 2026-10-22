module CorleoneZygoteExtension

import Zygote
import Corleone: is_extension_loaded, unwrapbuffer, wrapbuffer

is_extension_loaded(::Val{:Zygote}) = true

wrapbuffer(x) = Zygote.Buffer(x)
unwrapbuffer(x::Zygote.Buffer) = copy(x)

end

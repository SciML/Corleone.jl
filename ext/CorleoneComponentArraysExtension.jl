module CorleoneComponentArraysExtension
using Corleone
using ComponentArrays

Corleone.to_vec(::Val{:ComponentArrays}, u) = begin 
	collect(ComponentArray(u))
end

function Corleone.WrappedFunction(::Val{:ComponentArrays}, f, p, st; post, kwargs...) 
    u0 = ComponentVector(p)
	pre = let ax = getaxes(u0)
		(p) -> ComponentArray(p, ax) 	
	end
	Corleone.WrappedFunction{
		typeof(f), typeof(pre), typeof(post)
	}(f, pre, post)
end

end

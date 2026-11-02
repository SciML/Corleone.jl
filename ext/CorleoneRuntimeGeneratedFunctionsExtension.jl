module CorleoneRuntimeGeneratedFunctionsExtension

import RuntimeGeneratedFunctions
import Corleone: compile_evaluator

RuntimeGeneratedFunctions.init(@__MODULE__)

# Resolve globals explicitly so the user's module need not initialize RGF or be
# modified while this extension is being precompiled. The parser supplies one
# generated argument and a numerical expression body.
function expression_globals(expr::Expr, argument, context)
    return Expr(expr.head, map(x -> expression_globals(x, argument, context), expr.args)...)
end
expression_globals(sym::Symbol, argument, context) = sym === argument ? sym : GlobalRef(context, sym)
expression_globals(x, argument, context) = x

function compile_evaluator(::Val{:RuntimeGeneratedFunctions}, expression::Expr, eval_module::Module)
    argument, body = expression.args
    qualified = Expr(:->, argument, expression_globals(body, argument, eval_module))
    return RuntimeGeneratedFunctions.RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, qualified)
end

end

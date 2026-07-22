using SciMLTesting
using CorleoneOED
using Test

# These are the deliberate `@reexport using` facade APIs. Keep this snapshot explicit:
# additions are user-facing API changes and must be reviewed with their documentation.
const CORLEONE_REEXPORTS = (
    :ControlParameter, :Corleone, :CorleoneDynamicOptProblem, :MultipleShootingLayer,
    :SingleShootingLayer, :Trajectory, :constant_initialization, :custom_initialization,
    :default_initialization, :forward_initialization, :hybrid_initialization,
    :linear_initialization, :random_initialization,
)
const SYMBOLICS_REEXPORTS = (
    Symbol("@acrule"), Symbol("@arrayop"), Symbol("@derivative_rule"), Symbol("@makearray"),
    Symbol("@register_array_symbolic"), Symbol("@register_derivative"),
    Symbol("@register_discontinuity"), Symbol("@register_inverse"),
    Symbol("@register_symbolic"), Symbol("@rule"), Symbol("@symbolic_wrap"), Symbol("@syms"),
    Symbol("@symstruct"), Symbol("@variables"), Symbol("@wrapped"), :BS, :Differential,
    :Equation, :IRStructure, :Inequality, :Integral, :Num, :Rewriters, :RuleSet, :SafeReal,
    :SymReal, :SymStruct, :SymbolicLinearODE, :SymbolicUtils, :Symbolics,
    :SymbolicsSparsityDetector, :TreeReal, :approximation_function, :arguments,
    :build_function, :expand, :expand_derivatives, :factors, :flatten_fractions,
    :gather_factor, :get_canonical_expr, :get_reachability, :getmetadata, :groebner_basis,
    :has_inverse, :has_left_inverse, :has_right_inverse, :hasmetadata, :ifelse_branching,
    :ifelse_eager, :infimum, :inverse, :is_derivative, :is_groebner_basis, :iscall, :istree,
    :left_continuous_function, :left_inverse, :limit, :majorization_function,
    :minorization_function, :operation, :parse_expr_to_symbolic, :polynomial_coeffs,
    :populate_ir!, :print_ir, :quick_cancel, :right_continuous_function, :right_inverse,
    :rootfunction, :semilinear_form, :semipolynomial_form, :semiquadratic_form, :series,
    :setmetadata, :simplify, :simplify_fractions, :solve_for, :solve_linear_ode_system,
    :solve_symbolic_IVP, :sorted_arguments, :substitute, :substitute_in_deriv,
    :substitute_in_deriv_and_depvar, :supremum, :symbolic_linear_solve, :symbolic_solve,
    :symbolic_solve_ode, :symbolics_to_sympy, :symbolics_to_sympy_pythoncall,
    :sympy_algebraic_solve, :sympy_integrate, :sympy_limit, :sympy_linear_solve,
    :sympy_ode_solve, :sympy_pythoncall_algebraic_solve, :sympy_pythoncall_integrate,
    :sympy_pythoncall_limit, :sympy_pythoncall_linear_solve, :sympy_pythoncall_ode_solve,
    :sympy_pythoncall_simplify, :sympy_pythoncall_to_symbolics, :sympy_simplify,
    :sympy_to_symbolics, :taylor, :taylor_coeff, :term, :terms, :tosymbol, :unwrap_const,
    :vartype, Symbol("≲"), Symbol("≳"),
)
const SYMBOLICS_UNDOCUMENTED_REEXPORTS = (
    Symbol("@symbolic_wrap"), Symbol("@wrapped"), :RuleSet, :get_canonical_expr,
    :infimum, :is_derivative, :istree, :solve_for, :supremum,
)

run_qa(
    CorleoneOED;
    # CorleoneOED pulls Corleone and its other deps in with bare `using`, so it
    # leans on a large set of implicit imports. Converting every one to an
    # explicit import is a sizable refactor tracked in SciML/Corleone.jl#103.
    ei_broken = (:no_implicit_imports,),
    ei_kwargs = (;
        # Names still not declared public in their owning modules: SciMLBase
        # internals (`AbstractDEAlgorithm`, `AbstractDEProblem`, `get_colorizers`),
        # SciMLStructures internals (`Tunable`, `canonicalize`), SymbolicUtils
        # internal (`Code`), Symbolics internals (`getdefaultval`, `setdefaultval`,
        # `variables`), ForwardDiff internal (`jacobian`), and Corleone's own
        # as-yet-unexported helpers reached through `Corleone.*`.
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractDEAlgorithm, :AbstractDEProblem, :Code, :Tunable,
                :build_index_grid, :canonicalize, :get_block_structure,
                :get_bounds, :get_colorizers,
                :get_number_of_shooting_constraints, :get_timegrid, :getdefaultval,
                :jacobian, :retrieve_symbol_cache, :setdefaultval,
                :shooting_constraints, :shooting_constraints!, :variables,
            ),
        ),
    ),
    api_docs_kwargs = (;
        docs_src = normpath(@__DIR__, "..", "..", "..", "..", "docs", "src"),
        ignore = SYMBOLICS_UNDOCUMENTED_REEXPORTS,
        rendered_ignore = SYMBOLICS_REEXPORTS,
    ),
    reexports_allow = (CORLEONE_REEXPORTS..., SYMBOLICS_REEXPORTS...),
)

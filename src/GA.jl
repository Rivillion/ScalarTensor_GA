using DynamicExpressions
using SymbolicRegression
using Random
using ForwardDiff
using Symbolics

import SymbolicRegression.MutationFunctionsModule as SRMut
import SymbolicRegression: MutationWeights

const _HASHCONS_TABLES = (:WCS_SYMREAL, :WCS_SAFEREAL, :WCS_TREEREAL)

"""
    compact_symbolic_caches!() -> Bool

Drop dead entries from the global SymbolicUtils hash-consing tables. The tables keep a slot for every expression
ever built and only drop dead ones when they grow, so without this they grow by about 0.35 KB per simplified
individual (~1 GB over a 2000-generation run; docs/MEMORY_AUDIT.md). Call it once per generation, from the main
task, outside threaded sections; it takes the SymbolicUtils table lock. Live expressions are kept, so later
simplifications still find them.

Hash-consing itself must stay on: without it, Symbolics never finishes simplifying some trees.

Uses SymbolicUtils internals (`WCS_LOCK`, `WCS_*`, `WeakCacheSets.rehash!`). If they change in a package upgrade,
this warns once and does nothing. Returns whether it compacted.

Generated with Claude Opus 5.5.
"""
function compact_symbolic_caches!()
    SU = Symbolics.SymbolicUtils
    available = isdefined(SU, :WCS_LOCK) && all(name -> isdefined(SU, name), _HASHCONS_TABLES)
    if available
        tables = map(name -> getfield(SU, name), _HASHCONS_TABLES)
        available = all(tables) do table
            owner = parentmodule(typeof(table))
            isdefined(owner, :rehash!) && hasproperty(table, :count)
        end
    end
    if !available
        @warn "SymbolicUtils internals changed; hash-consing tables are not compacted" maxlog = 1
        return false
    end
    @lock SU.WCS_LOCK for table in tables
        # `count` includes dead entries, so 1.5 × count always leaves room for every live one.
        parentmodule(typeof(table)).rehash!(table, max(16, cld(3 * table.count, 2)))
    end
    return true
end

# Operators used by SymbolicRegression
const sr_options = Options(
    binary_operators=[+, -, *],
    unary_operators=[exp]
)

const operators = OperatorEnum(
    binary_operators=(+, -, *),
    unary_operators=(exp,)
)


# Structure for individuals
struct Individual
    F::Node{Float64}
    U::Node{Float64}
    birth_gen::Int
    complexity::Int
end

"""An immutable, user-facing description of one physical `(F(Phi), U(Phi))` proposal."""
Base.@kwdef struct GuessSpec
    label::String
    F::String
    U::String
end

"""Configuration for initial physical seeding and optional stagnation reinjection."""
Base.@kwdef struct SeedSettings
    guesses::Vector{GuessSpec} = GuessSpec[]
    initial_fraction::Float64 = 0.03
    neighbor_mutations_min::Int = 1
    neighbor_mutations_max::Int = 3
    mutation_retry_limit::Int = 20
    require_function_sanity::Bool = true
    reject_duplicate_guesses::Bool = true
    reinject_after::Union{Nothing,Int} = nothing
    reinject_interval::Int = 25
    reinjection_fraction::Float64 = 0.01
end

Base.@kwdef struct PenaltySettings
    mode::Symbol = :functional_only
    fisher_pruning::Bool = true
end

function validate_penalty_settings(settings::PenaltySettings)
    settings.mode in (:functional_only, :all_coefficients) || throw(ArgumentError(
        "penalty mode must be :functional_only or :all_coefficients; got $(settings.mode)"))
    settings.mode == :functional_only && settings.fisher_pruning && return nothing
    return nothing
end


# Complexity required for parsimony penalty
@inline function compute_complexity(F::Node, U::Node)
    return count_nodes(F) + count_nodes(U)
end

# Check tree constraints
@inline function check_tree_constraints(tree::Node, curmaxsize::Int, maxdepth::Int)
    return count_nodes(tree) <= curmaxsize && count_depth(tree) <= maxdepth
end

"""
    parse_guess_tree(expression)

Parse a physical-guess expression using the GA's operator registry and the single,
case-sensitive variable `Phi`. The returned tree never aliases parser-owned state.

Generated with GPT 5.6.
"""
function parse_guess_tree(expression::AbstractString)::Node{Float64}
    source = String(expression)
    try
        parsed = parse_expression(
            Meta.parse(source);
            operators=sr_options.operators,
            variable_names=["Phi"],
            node_type=Node{Float64},
            expression_type=Expression,
        )
        tree = copy_node(get_tree(parsed))
        is_tree_finite(tree) || throw(ArgumentError("expression contains a non-finite constant"))
        return tree
    catch error
        reason = sprint(showerror, error)
        throw(ArgumentError("invalid physical guess expression `$(source)`: $(reason)"))
    end
end

"""Return a stable, human-readable signature for an individual's current trees.

Generated with GPT 5.6.
"""
function individual_signature(ind::Individual)::Tuple{String,String}
    return (
        string_tree(ind.F, operators; variable_names=["Phi"]),
        string_tree(ind.U, operators; variable_names=["Phi"]),
    )
end

"""Copy an individual so its mutable trees are independently owned.

Generated with GPT 5.6.
"""
@inline function _copy_individual(ind::Individual, generation::Int=ind.birth_gen)
    F = copy_node(ind.F)
    U = copy_node(ind.U)
    return Individual(F, U, generation, compute_complexity(F, U))
end

"""Warn when a nonconstant guess has no currently optimizable coefficients.

Generated with GPT 5.6.
"""
function _warn_integer_only_coefficients(guess::GuessSpec, ind::Individual)
    nodes = Iterators.flatten((ind.F, ind.U))
    has_variable = any(node -> node.degree == 0 && !node.constant, nodes)
    constants = [node.val for tree in (ind.F, ind.U) for node in tree
        if node.degree == 0 && node.constant]
    if has_variable && !isempty(constants) && all(isinteger, constants)
        @warn "Physical guess has no optimizable coefficients: integer-valued constants are structural under the current rule" label=guess.label
    end
    return nothing
end

"""Construct, simplify, and validate one exact physical seed.

Generated with GPT 5.6.
"""
function individual_from_guess(
    guess::GuessSpec,
    generation::Int,
    max_size::Int;
    require_function_sanity::Bool,
)::Individual
    F_tree = try
        parse_guess_tree(guess.F)
    catch error
        throw(ArgumentError("invalid F expression for guess `$(guess.label)` (`$(guess.F)`): " *
            sprint(showerror, error)))
    end
    U_tree = try
        parse_guess_tree(guess.U)
    catch error
        throw(ArgumentError("invalid U expression for guess `$(guess.label)` (`$(guess.U)`): " *
            sprint(showerror, error)))
    end

    raw = Individual(F_tree, U_tree, generation, compute_complexity(F_tree, U_tree))
    simplified = simplify_individual(raw)
    F = copy_node(simplified.F)
    U = copy_node(simplified.U)
    ind = Individual(F, U, generation, compute_complexity(F, U))

    is_tree_finite(ind.F) || throw(ArgumentError(
        "guess `$(guess.label)` has non-finite constants in F (`$(guess.F)`)"))
    is_tree_finite(ind.U) || throw(ArgumentError(
        "guess `$(guess.label)` has non-finite constants in U (`$(guess.U)`)"))
    check_tree_constraints(ind.F, max_size, sr_options.maxdepth) || throw(ArgumentError(
        "guess `$(guess.label)` F (`$(guess.F)`) exceeds max_size=$max_size or maxdepth=$(sr_options.maxdepth)"))
    check_tree_constraints(ind.U, max_size, sr_options.maxdepth) || throw(ArgumentError(
        "guess `$(guess.label)` U (`$(guess.U)`) exceeds max_size=$max_size or maxdepth=$(sr_options.maxdepth)"))
    if require_function_sanity && !passes_function_sanity(ind)
        throw(ArgumentError("guess `$(guess.label)` fails function sanity checks " *
            "for F=`$(guess.F)`, U=`$(guess.U)`"))
    end
    _warn_integer_only_coefficients(guess, ind)
    return _copy_individual(ind)
end

"""Build all exact seeds in configuration order, rejecting ambiguous proposals.

Generated with GPT 5.6.
"""
function build_exact_seeds(
    guesses::Vector{GuessSpec},
    generation::Int,
    max_size::Int,
    settings::SeedSettings,
)::Vector{Individual}
    labels = Dict{String,Int}()
    signatures = Dict{Tuple{String,String},Int}()
    seeds = Individual[]
    for (index, guess) in pairs(guesses)
        isempty(strip(guess.label)) && throw(ArgumentError("physical guess $index has an empty label"))
        if haskey(labels, guess.label)
            other = guesses[labels[guess.label]].label
            throw(ArgumentError("duplicate physical guess label `$(guess.label)` (also used by `$other`)"))
        end
        labels[guess.label] = index
        seed = individual_from_guess(guess, generation, max_size;
            require_function_sanity=settings.require_function_sanity)
        signature = individual_signature(seed)
        if haskey(signatures, signature)
            other = guesses[signatures[signature]].label
            message = "physical guesses `$other` and `$(guess.label)` simplify to the same (F, U) pair"
            settings.reject_duplicate_guesses ? throw(ArgumentError(message)) : @warn(message)
        else
            signatures[signature] = index
        end
        push!(seeds, _copy_individual(seed))
    end
    return seeds
end


# Auto-diff evaluator required for cosmology
@inline function eval_tree_scalar(tree::Node{Float64}, x::T) where {T}
    if tree.degree == 0
        return tree.constant ? T(tree.val) : x
    elseif tree.degree == 1
        op = tree.op
        child = eval_tree_scalar(tree.l, x)

        # Unary Ops Map
        op == 1 && return exp(child)
    else
        op = tree.op
        left = eval_tree_scalar(tree.l, x)
        right = eval_tree_scalar(tree.r, x)

        # Binary Ops Map
        op == 1 && return left + right
        op == 2 && return left - right
        op == 3 && return left * right
    end
    return T(NaN)
end

function build_F_funcs(tree::Node{Float64})
    return Φ -> eval_tree_value_derivatives(tree, Φ)
end

function build_U_funcs(tree::Node{Float64})
    return Φ -> begin
        val, deriv, _ = eval_tree_value_derivatives(tree, Φ)
        (val, deriv)
    end
end

@inline function eval_tree_value_derivatives(tree::Node{Float64}, x::T) where {T}
    if tree.degree == 0
        return tree.constant ? (T(tree.val), zero(x), zero(x)) : (x, one(x), zero(x))
    elseif tree.degree == 1
        val, deriv, deriv2 = eval_tree_value_derivatives(tree.l, x)

        if tree.op == 1
            exp_val = exp(val)
            return exp_val, exp_val * deriv, exp_val * (deriv * deriv + deriv2)
        end
    else
        left_val, left_deriv, left_deriv2 = eval_tree_value_derivatives(tree.l, x)
        right_val, right_deriv, right_deriv2 = eval_tree_value_derivatives(tree.r, x)

        if tree.op == 1
            return left_val + right_val, left_deriv + right_deriv, left_deriv2 + right_deriv2
        elseif tree.op == 2
            return left_val - right_val, left_deriv - right_deriv, left_deriv2 - right_deriv2
        elseif tree.op == 3
            return (
                left_val * right_val,
                left_deriv * right_val + left_val * right_deriv,
                left_deriv2 * right_val + 2.0 * left_deriv * right_deriv + left_val * right_deriv2,
            )
        end
    end

    nan_val = T(NaN)
    return nan_val, nan_val, nan_val
end

@inline function eval_tree_value_derivatives(tree::Node{Float64}, x::Float64)
    if tree.degree == 0
        return tree.constant ? (tree.val, 0.0, 0.0) : (x, 1.0, 0.0)
    elseif tree.degree == 1
        val, deriv, deriv2 = eval_tree_value_derivatives(tree.l, x)

        if tree.op == 1
            exp_val = exp(val)
            return exp_val, exp_val * deriv, exp_val * (deriv * deriv + deriv2)
        end
    else
        left_val, left_deriv, left_deriv2 = eval_tree_value_derivatives(tree.l, x)
        right_val, right_deriv, right_deriv2 = eval_tree_value_derivatives(tree.r, x)

        if tree.op == 1
            return left_val + right_val, left_deriv + right_deriv, left_deriv2 + right_deriv2
        elseif tree.op == 2
            return left_val - right_val, left_deriv - right_deriv, left_deriv2 - right_deriv2
        elseif tree.op == 3
            return (
                left_val * right_val,
                left_deriv * right_val + left_val * right_deriv,
                left_deriv2 * right_val + 2.0 * left_deriv * right_deriv + left_val * right_deriv2,
            )
        end
    end

    return NaN, NaN, NaN
end

# LCDM Test Case
function compile_tree_lcdm()
    F_tree = Node(Float64; val=1.0)
    U_tree = Node(Float64; val=0.0)
    return build_F_funcs(F_tree), build_U_funcs(U_tree)
end

function to_symbolics(tree::Node{Float64}, Φ::Num)
    if tree.degree == 0
        return tree.constant ? tree.val : Φ
    elseif tree.degree == 1
        op = tree.op
        child = to_symbolics(tree.l, Φ)
        op == 1 && return exp(child)
    elseif tree.degree == 2
        op = tree.op
        left = to_symbolics(tree.l, Φ)
        right = to_symbolics(tree.r, Φ)
        op == 1 && return left + right
        op == 2 && return left - right
        op == 3 && return left * right
    end
    return NaN
end


# =========================================
# Genetic operators from SymbolicRegression
# =========================================

# Old simplification based on DynamicsExpressions
# This only compares adjacent nodes and leafs, and so it misses very elemental simplifications
# that are very simple at first glance but convoluted in a tree representation
# function simplify_individual(ind::Individual, options::Options)
#     ```
#     Simplify constants in functions before giving them to the optimizer
#     ```
#     F_copy = copy_node(ind.F)
#     U_copy = copy_node(ind.U)

#     simplified_F = combine_operators(simplify_tree!(F_copy, options.operators), options.operators)
#     simplified_U = combine_operators(simplify_tree!(U_copy, options.operators), options.operators)

#     return Individual(simplified_F, simplified_U, ind.birth_gen, compute_complexity(simplified_F, simplified_U))
# end

# Updated simplification based on SymbolicUtils
# We convert the tree into a symbolic expression, perform the simplification,
# and return back to a DynamicExpressions function
# function rebuild_canonical(expr)
#     # ```
#     # Avoids weird bugs with simplification when converting trees into symbolic expressions.
#     # Example: x - (c1*x) from a tree will not be simplified 
#     #          when using node_to_symbolic but x + (-c1*x) will.
#     # ```
#     if istree(expr)
#         op = operation(expr)
#         # Recursively canonicalize the arguments first (bottom-up)
#         args = map(rebuild_canonical, arguments(expr))
#         # Collapse math correctly for SymbolicUtils
#         return op(args...)
#     else
#         # Don't do anything with constants and variables
#         return expr
#     end
# end

function restore_subtractions!(tree::Node{Float64})
    # ```
    # Avoids weird bugs with simplification when converting symbolic expressions into trees
    # Example: exp(x) - x is converted into exp + (-1.0 * x), adding extra (unfair) complexity
    # ```
    if tree.degree == 0
        return tree

        # Unary operator
    elseif tree.degree == 1
        tree.l = restore_subtractions!(tree.l)
        return tree

        # Binary operator
    elseif tree.degree == 2
        # Bottom-up recursion: clean the children first
        tree.l = restore_subtractions!(tree.l)
        tree.r = restore_subtractions!(tree.r)

        # Look for the "+" operator (Index 1)
        if tree.op == 1

            # Case 1: Right child is a multiplication by -1.0
            if tree.r.degree == 2 && tree.r.op == 3 # 3 is *
                if tree.r.l.degree == 0 && tree.r.l.constant && tree.r.l.val == -1.0
                    # A + (-1.0 * B) -> A - B
                    tree.op = 2 # Change operator to - (Index 2)
                    tree.r = tree.r.r # Bypass the multiplication node
                    return tree
                elseif tree.r.r.degree == 0 && tree.r.r.constant && tree.r.r.val == -1.0
                    # A + (B * -1.0) -> A - B
                    tree.op = 2
                    tree.r = tree.r.l
                    return tree
                end
            end

            # Case 2: Left child is a multiplication by -1.0
            if tree.l.degree == 2 && tree.l.op == 3
                if tree.l.l.degree == 0 && tree.l.l.constant && tree.l.l.val == -1.0
                    # (-1.0 * B) + A -> A - B
                    tree.op = 2
                    new_r = tree.l.r  # B becomes the right child
                    tree.l = tree.r   # A becomes the left child
                    tree.r = new_r
                    return tree
                elseif tree.l.r.degree == 0 && tree.l.r.constant && tree.l.r.val == -1.0
                    # (B * -1.0) + A -> A - B
                    tree.op = 2
                    new_r = tree.l.l
                    tree.l = tree.r
                    tree.r = new_r
                    return tree
                end
            end
        end
    end

    return tree
end

function restore_subtractions!(tree::Node{Float32})
    # ```
    # Avoids weird bugs with simplification when converting symbolic expressions into trees
    # Example: exp(x) - x is converted into exp + (-1.0 * x), adding extra (unfair) complexity
    # ```
    if tree.degree == 0
        return tree

        # Unary operator
    elseif tree.degree == 1
        tree.l = restore_subtractions!(tree.l)
        return tree

        # Binary operator
    elseif tree.degree == 2
        # Bottom-up recursion: clean the children first
        tree.l = restore_subtractions!(tree.l)
        tree.r = restore_subtractions!(tree.r)

        # Look for the "+" operator (Index 1)
        if tree.op == 1

            # Case 1: Right child is a multiplication by -1.0
            if tree.r.degree == 2 && tree.r.op == 3 # 3 is *
                if tree.r.l.degree == 0 && tree.r.l.constant && tree.r.l.val == -1.0
                    # A + (-1.0 * B) -> A - B
                    tree.op = 2 # Change operator to - (Index 2)
                    tree.r = tree.r.r # Bypass the multiplication node
                    return tree
                elseif tree.r.r.degree == 0 && tree.r.r.constant && tree.r.r.val == -1.0
                    # A + (B * -1.0) -> A - B
                    tree.op = 2
                    tree.r = tree.r.l
                    return tree
                end
            end

            # Case 2: Left child is a multiplication by -1.0
            if tree.l.degree == 2 && tree.l.op == 3
                if tree.l.l.degree == 0 && tree.l.l.constant && tree.l.l.val == -1.0
                    # (-1.0 * B) + A -> A - B
                    tree.op = 2
                    new_r = tree.l.r  # B becomes the right child
                    tree.l = tree.r   # A becomes the left child
                    tree.r = new_r
                    return tree
                elseif tree.l.r.degree == 0 && tree.l.r.constant && tree.l.r.val == -1.0
                    # (B * -1.0) + A -> A - B
                    tree.op = 2
                    new_r = tree.l.l
                    tree.l = tree.r
                    tree.r = new_r
                    return tree
                end
            end
        end
    end

    return tree
end

function is_tree_finite(tree::Node{Float64})
    for node in tree
        if node.degree == 0 && node.constant && !isfinite(node.val)
            return false
        end
    end
    return true
end

function is_tree_finite(tree::Node{Float32})
    for node in tree
        if node.degree == 0 && node.constant && !isfinite(node.val)
            return false
        end
    end
    return true
end

function simplify_tree_custom(tree::Node{Float64})
    ```
    Simplify the algebra and the constants of a function before optimization.
    ```
    tree_copy = copy_node(tree)

    # Convert to symbolic
    try
        raw_sym_tree = node_to_symbolic(tree_copy, operators)
        # Expand and simplify
        sym_tree = simplify(raw_sym_tree; expand=true)
        simplified_tree = sym_tree isa Number ? Node(Float64; val=Float64(sym_tree)) : symbolic_to_node(sym_tree, operators)
        simplified_tree = restore_subtractions!(simplified_tree)

        # Check if everything is finite
        if !is_tree_finite(simplified_tree)
            return tree_copy
        end

        return simplified_tree

    catch
        return tree_copy
    end
end

# function simplify_individual(ind::Individual)
#     ```
#     Simplify the algebra and the constants before optimization.
#     ```
#     F_copy = copy_node(ind.F)
#     U_copy = copy_node(ind.U)

#     # Convert to symbolic
#     raw_sym_F = node_to_symbolic(F_copy, operators)
#     raw_sym_U = node_to_symbolic(U_copy, operators)

#     # Collapse math correctly before simplification
#     # For convoluted expressions, expand can help to reduce constants
#     # e.g. a+(x+b)(x+c) can be reduced to a + b*x + x²
#     # sym_F = simplify(rebuild_canonical(raw_sym_F); expand=true)
#     # sym_U = simplify(rebuild_canonical(raw_sym_U); expand=true)
#     sym_F = simplify(raw_sym_F; expand=true)
#     sym_U = simplify(raw_sym_U; expand=true)

#     # Simplify
#     # If we have a pure constant, we do not give it directly to symbolic_to_node to avoid an error
#     simplified_F = sym_F isa Number ? Node(Float64; val=Float64(sym_F)) : symbolic_to_node(sym_F, operators)
#     simplified_U = sym_U isa Number ? Node(Float64; val=Float64(sym_U)) : symbolic_to_node(sym_U, operators)

#     simplified_F = restore_subtractions!(simplified_F)
#     simplified_U = restore_subtractions!(simplified_U)

#     # Check if simplified tree is not
#     if !is_tree_finite(simplified_F) || !is_tree_finite(simplified_U)
#         return Individual(F_copy, U_copy, ind.birth_gen, compute_complexity(F_copy, U_copy))
#     end

#     return Individual(combine_operators(simplify_tree!(copy(simplified_F), operators), operators),
#         combine_operators(simplify_tree!(copy(simplified_U), operators), operators),
#         ind.birth_gen, compute_complexity(simplified_F, simplified_U))
# end

function simplify_individual(ind::Individual)
    ```
    Simplify the algebra and the constants before optimization.
    ```
    F_copy = copy_node(ind.F)
    U_copy = copy_node(ind.U)

    # Simplified trees
    simplified_F = simplify_tree_custom(F_copy)
    simplified_U = simplify_tree_custom(U_copy)

    F = combine_operators(simplify_tree!(copy(simplified_F), operators), operators)
    U = combine_operators(simplify_tree!(copy(simplified_U), operators), operators)

    # The final SymbolicRegression pass can change the number of nodes, so the
    # complexity must be computed from the trees that are actually returned.
    return Individual(F, U, ind.birth_gen, compute_complexity(F, U))
end

function generate_random_individual(rng::AbstractRNG, max_nodes::Int, gen::Int)
    ```
    Generates a random individual as in SymbolicExpressions.
    We do it up to a maximum, user selected, number of nodes.
    ```
    # nfeatures = 1 (i.e. we only use the field Φ)
    F_size = rand(rng, 1:max_nodes)
    U_size = rand(rng, 1:max_nodes)

    F_tree = SRMut.gen_random_tree_fixed_size(F_size, sr_options, 1, Float64, rng)
    U_tree = SRMut.gen_random_tree_fixed_size(U_size, sr_options, 1, Float64, rng)

    return Individual(F_tree, U_tree, gen, compute_complexity(F_tree, U_tree))
end

generate_random_individual(max_nodes::Int, gen::Int) =
    generate_random_individual(Random.default_rng(), max_nodes, gen)

"""
    crossover_sr(ind1::Individual, ind2::Individual, current_gen::Int, curmaxsize::Int=15)

Implements the crossover as in SymbolicExpressions.
"""
function crossover_sr(rng::AbstractRNG, ind1::Individual, ind2::Individual, current_gen::Int,
    curmaxsize::Int=15)
    # Pick either F or U to crossover
    cross_F = rand(rng) < 0.5
    t1 = cross_F ? ind1.F : ind1.U
    t2 = cross_F ? ind2.F : ind2.U

    # If we select the exact same trees, just copy
    if t1 === t2
        return Individual(copy_node(ind1.F), copy_node(ind1.U), current_gen, ind1.complexity),
        Individual(copy_node(ind2.F), copy_node(ind2.U), current_gen, ind2.complexity)
    end

    # Attempt crossover max_attempts times, checking constraints 
    # (max depth and max complexity) are satisfied
    max_attempts = 10
    for attempt in 1:max_attempts
        c1, c2 = SRMut.crossover_trees(t1, t2, rng)

        # Check constraints
        if check_tree_constraints(c1, curmaxsize, sr_options.maxdepth) &&
           check_tree_constraints(c2, curmaxsize, sr_options.maxdepth)
            if cross_F
                return Individual(c1, copy_node(ind1.U), current_gen, compute_complexity(c1, ind1.U)),
                Individual(c2, copy_node(ind2.U), current_gen, compute_complexity(c2, ind2.U))
            else
                return Individual(copy_node(ind1.F), c1, current_gen, compute_complexity(ind1.F, c1)),
                Individual(copy_node(ind2.F), c2, current_gen, compute_complexity(ind2.F, c2))
            end
        end
    end

    # If all attempts failed, return original individuals
    return Individual(copy_node(ind1.F), copy_node(ind1.U), current_gen, ind1.complexity),
    Individual(copy_node(ind2.F), copy_node(ind2.U), current_gen, ind2.complexity)
end


crossover_sr(ind1::Individual, ind2::Individual, current_gen::Int, curmaxsize::Int=15) =
    crossover_sr(Random.default_rng(), ind1, ind2, current_gen, curmaxsize)

function sample_mutation_rng(rng::AbstractRNG, weights)
    names = fieldnames(typeof(weights))
    values = Float64[getfield(weights, name) for name in names]
    total = sum(values)
    total > 0.0 || return :do_nothing
    target = rand(rng) * total
    cumulative = 0.0
    for (name, value) in zip(names, values)
        cumulative += value
        target <= cumulative && return name
    end
    return names[end]
end


"""
    mutate_sr(ind::Individual, current_gen::Int, curmaxsize::Int=15)

Implement the mutation as in SymbolicRegressions.
"""
function mutate_sr(rng::AbstractRNG, ind::Individual, current_gen::Int, curmaxsize::Int=15)
    # Pick either F or U to mutate, and the particular mutation type
    mutate_F = rand(rng) < 0.5
    tree = mutate_F ? copy_node(ind.F) : copy_node(ind.U)
    n = count_nodes(tree)
    depth = count_depth(tree)

    # Analyze the tree to adjust the mutation weights
    weights = copy(sr_options.mutation_weights)
    weights.form_connection = 0.0
    weights.break_connection = 0.0
    if hasfield(typeof(weights), :mutate_feature)
        weights.mutate_feature = 0.0
    end

    # If we just have a constant or a variable, omit it some operations
    if tree.degree == 0
        weights.mutate_operator = 0.0
        weights.swap_operands = 0.0
        weights.delete_node = 0.0
        weights.simplify = 0.0
        if !tree.constant
            weights.optimize = 0.0
            weights.mutate_constant = 0.0
        end
    else
        # More constants -> Higher chance of constant mutation
        weights.mutate_constant *= min(8, count_constants(tree)) / 8.0
    end

    # If there are no binary operations, we cannot swap
    if !any(node -> node.degree == 2, tree)
        weights.swap_operands = 0.0
    end

    if n >= curmaxsize || depth >= sr_options.maxdepth
        weights.add_node = 0.0
        weights.insert_node = 0.0
    end

    # Select the mutation,and attempt it max_attempts times,
    # checking (max depth and max complexity) are satisfied
    mutation_choice = sample_mutation_rng(rng, weights)
    max_attempts = 10
    for attempt in 1:max_attempts
        if mutation_choice == :mutate_constant
            tree = SRMut.mutate_constant(tree, 0.5, sr_options, rng) # Temperature T=0.5

        elseif mutation_choice == :mutate_operator
            tree = SRMut.mutate_operator(tree, sr_options, rng)

        elseif mutation_choice == :swap_operands
            tree = SRMut.swap_operands(tree, rng)

        elseif mutation_choice == :rotate_tree
            tree = SRMut.randomly_rotate_tree!(tree, rng)

        elseif mutation_choice == :add_node
            if rand(rng) < 0.5
                # Add random unary/binary operation to the end of a tree
                tree = SRMut.append_random_op(tree, sr_options, 1, rng)
            else
                # Add random node to the top of a tree
                tree = SRMut.prepend_random_op(tree, sr_options, 1, rng)
            end

        elseif mutation_choice == :insert_node
            tree = SRMut.insert_random_op(tree, sr_options, 1, rng)

        elseif mutation_choice == :delete_node
            tree = SRMut.delete_random_op!(tree, sr_options, 1, rng)

        elseif mutation_choice == :simplify
            simplify_tree!(tree, sr_options.operators)
            tree = combine_operators(tree, sr_options.operators)

        elseif mutation_choice == :randomize
            tree = SRMut.randomize_tree(tree, curmaxsize, sr_options, 1, rng)

        elseif mutation_choice == :optimize
            # We skip it since we optimize constants always anyway

        elseif mutation_choice == :do_nothing
            # Skip entirely

        end

        # Return mutated within the Individual struct
        if check_tree_constraints(tree, curmaxsize, sr_options.maxdepth)
            if mutate_F
                return Individual(tree, copy_node(ind.U), current_gen, compute_complexity(tree, ind.U))
            else
                return Individual(copy_node(ind.F), tree, current_gen, compute_complexity(ind.F, tree))
            end
        end
    end

    # If all mutation attempts have failed, return the unchanged original individual
    return ind
end


mutate_sr(ind::Individual, current_gen::Int, curmaxsize::Int=15) =
    mutate_sr(Random.default_rng(), ind, current_gen, curmaxsize)

"""Generate one valid, unique mutated descendant of an exact physical seed.

Generated with GPT 5.6.
"""
function mutated_seed_variant(
    rng::AbstractRNG,
    seed::Individual,
    generation::Int,
    max_size::Int,
    settings::SeedSettings,
    forbidden_signatures::Set{Tuple{String,String}},
)::Union{Individual,Nothing}
    source_signature = individual_signature(seed)
    for _ in 1:settings.mutation_retry_limit
        variant = _copy_individual(seed, generation)
        mutation_steps = rand(rng,
            settings.neighbor_mutations_min:settings.neighbor_mutations_max)
        for _ in 1:mutation_steps
            variant = mutate_sr(rng, variant, generation, max_size)
        end
        variant = simplify_individual(variant)
        F = copy_node(variant.F)
        U = copy_node(variant.U)
        variant = Individual(F, U, generation, compute_complexity(F, U))
        check_tree_constraints(variant.F, max_size, sr_options.maxdepth) || continue
        check_tree_constraints(variant.U, max_size, sr_options.maxdepth) || continue
        is_tree_finite(variant.F) && is_tree_finite(variant.U) || continue
        settings.require_function_sanity && !passes_function_sanity(variant) && continue
        signature = individual_signature(variant)
        signature == source_signature && continue
        signature in forbidden_signatures && continue
        return _copy_individual(variant)
    end
    return nothing
end

"""Return whether physical-seed reinjection is due at this stagnation count.

Generated with GPT 5.6.
"""
function seed_reinjection_due(
    generations_since_improvement::Int,
    settings::SeedSettings,
)::Bool
    threshold = settings.reinject_after
    return threshold !== nothing &&
        generations_since_improvement >= threshold &&
        (generations_since_improvement - threshold) % settings.reinject_interval == 0 &&
        settings.reinjection_fraction > 0.0 &&
        !isempty(settings.guesses)
end

"""Replace distinct child slots with unique mutated physical-seed descendants.

Generated with GPT 5.6.
"""
function inject_seed_variants!(
    rng::AbstractRNG,
    children::Vector{Individual},
    exact_seeds::Vector{Individual},
    generation::Int,
    max_size::Int,
    settings::SeedSettings,
)::Int
    isempty(children) && return 0
    isempty(exact_seeds) && return 0
    requested = clamp(round(Int, settings.reinjection_fraction * length(children)),
        0, length(children))
    requested == 0 && return 0
    positions = randperm(rng, length(children))[1:requested]
    forbidden = Set(individual_signature(seed) for seed in exact_seeds)
    union!(forbidden, (individual_signature(child) for child in children))
    injected = 0
    for (offset, position) in pairs(positions)
        seed = exact_seeds[mod1(offset, length(exact_seeds))]
        variant = mutated_seed_variant(rng, seed, generation, max_size, settings, forbidden)
        variant === nothing && continue
        children[position] = _copy_individual(variant, generation)
        push!(forbidden, individual_signature(variant))
        injected += 1
    end
    return injected
end


# ====================================
# Minimum description length principle
# ====================================
"""
    get_complete_U(u_tree::Node{Float64})

Constructs the complete U=U₀+u function to correctly account for the complexity of U0+ when needed.
This allows to search/decouple u to find it in the GA.
"""
@inline function get_complete_U(u_tree::Node{Float64})
    # We add a tiny value to avoid simplifications such as 0+Phi=Phi
    # But at the same time, we allow for simplifications such as c1+c2=c3
    return Node(1, Node(Float64; val=1e-15), copy_node(u_tree))
end


"""
    extract_unique_operators(tree::Node{Float64})

Counts the unique elements of the basis that compose a function.
Repeated variables or constants are not included in the counting.
Integer constants and real ones are not double counted. Matches ESR.
"""
function extract_unique_operators(tree::Node{Float64})
    # Using Sets to automatically keep only unique values
    nullary_ops = Set{Symbol}()
    unary_ops = Set{UInt8}()
    binary_ops = Set{UInt8}()

    # Iterate directly over the raw tree's nodes
    for node in tree
        if node.degree == 0
            if node.constant
                push!(nullary_ops, :constant)
            else
                push!(nullary_ops, Symbol("feature_$(node.feature)"))
            end
        elseif node.degree == 1
            push!(unary_ops, node.op)
        elseif node.degree == 2
            push!(binary_ops, node.op)
        end
    end

    # Return total unique elements
    return length(nullary_ops) + length(unary_ops) + length(binary_ops)
end


"""
    extract_unique_operators(tree::Node{Float32})

Counts the unique elements of the basis that compose a function.
Repeated variables or constants are not included in the counting.
Integer constants and real ones are not double counted. Matches ESR.
"""
function extract_unique_operators(tree::Node{Float32})
    # Using Sets to automatically keep only unique values
    nullary_ops = Set{Symbol}()
    unary_ops = Set{UInt8}()
    binary_ops = Set{UInt8}()

    # Iterate directly over the raw tree's nodes
    for node in tree
        if node.degree == 0
            if node.constant
                push!(nullary_ops, :constant)
            else
                push!(nullary_ops, Symbol("feature_$(node.feature)"))
            end
        elseif node.degree == 1
            push!(unary_ops, node.op)
        elseif node.degree == 2
            push!(binary_ops, node.op)
        end
    end

    # Return total unique elements
    return length(nullary_ops) + length(unary_ops) + length(binary_ops)
end



function functional_complexity(F_tree::Node{Float64}, U_tree::Node{Float64})
    ```
    We include the penalization term k log n, where:
    k : number of nodes
    n : unique operators obtained from extract_unique_operators
    Since we have two functions, we do this for both and sum
    ```
    # Functional complexities
    n_F = extract_unique_operators(F_tree)
    k_F = count_nodes(F_tree)

    n_U = extract_unique_operators(U_tree)
    k_U = count_nodes(U_tree)

    return k_F * log(n_F) + k_U * log(n_U)
end

function functional_complexity(F_tree::Node{Float64}, U_tree::Node{Float32})
    ```
    We include the penalization term k log n, where:
    k : number of nodes
    n : unique operators obtained from extract_unique_operators
    Since we have two functions, we do this for both and sum
    ```
    # Functional complexities
    n_F = extract_unique_operators(F_tree)
    k_F = count_nodes(F_tree)

    n_U = extract_unique_operators(U_tree)
    k_U = count_nodes(U_tree)

    return k_F * log(n_F) + k_U * log(n_U)
end



function structural_complexity(F_tree::Node{Float64}, U_tree::Node{Float64})
    ```
    Obtain the term log abs c_j from both functions.
    ```
    c_j = vcat(get_structural_constants(F_tree), get_structural_constants(U_tree))
    valid_c = filter(c -> c > abs(1.0), abs.(c_j))
    return isempty(valid_c) ? 0.0 : sum(log.(valid_c))
end

function structural_complexity(F_tree::Node{Float64}, U_tree::Node{Float32})
    ```
    Obtain the term log abs c_j from both functions.
    ```
    c_j = vcat(get_structural_constants(F_tree), get_structural_constants(U_tree))
    valid_c = filter(c -> c > abs(1.0), abs.(c_j))
    return isempty(valid_c) ? 0.0 : sum(log.(valid_c))
end

function get_structural_constants(tree::Node{Float64})
    ```
    Obtain constants arising from simplification.
    ```
    consts = Float64[]
    for node in tree
        if node.degree == 0 && node.constant && isinteger(node.val)
            push!(consts, node.val)
        end
    end
    return consts
end

function get_structural_constants(tree::Node{Float32})
    ```
    Obtain constants arising from simplification.
    ```
    consts = Float32[]
    for node in tree
        if node.degree == 0 && node.constant && isinteger(node.val)
            push!(consts, node.val)
        end
    end
    return consts
end

function fisher_complexity(fisher_diags::AbstractVector{Float64})
    isempty(fisher_diags) && return 0.0

    if any(d -> !isfinite(d) || d <=0, fisher_diags)
        return 1e8
    end

    return 0.5 * sum(log.(fisher_diags))
end

"""
Compute the Fisher-complexity contribution using the explicit Fisher availability status.
Invalid Fisher states receive the rejection penalty, while `:not_applicable` contributes zero.

Generated with GPT-5.
"""
function fisher_complexity(fisher_diags::AbstractVector{Float64}, fisher_status::Symbol)
    fisher_status == :not_applicable && return 0.0
    fisher_status == :valid && return fisher_complexity(fisher_diags)
    return 1e8
end


function penalty_components(ind::Individual, fisher_diags::AbstractVector{Float64}=Float64[];
    fisher_status::Symbol=:valid, settings::PenaltySettings=PenaltySettings())
    ```
    Compute the Minimum Description Length term for penalization.
    We should take into account the individual with its two functions.
    ```
    # We first create the complete U(Φ)=U₀+u(Φ) function.
    # We also apply an additional simplification.
    # If the u(Φ) ends up being a constant, we would have c1 + c2 and penalize more than necessary
    complete_F = ind.F
    # The Symbolics round-trip below cost ~433 KiB and ~1 ms per fitted child. The penalty total is disabled
    # (see `total` below), so these parts are diagnostics only; the cheap DynamicExpressions pass still merges
    # U0 with a constant term of u. Values match the Symbolics version on all stored candidates (functional) and
    # 99.4% (structural); see docs/MEMORY_AUDIT.md. Restore the Symbolics line if the penalty is turned back on.
    # complete_U = simplify_tree_custom(copy(get_complete_U(ind.U)))
    complete_U = combine_operators(simplify_tree!(get_complete_U(ind.U), operators), operators)


    # 1. Functional complexity k * log(n)
    functional_pen = functional_complexity(complete_F, complete_U)

    # 2. Structural penalty sum_j log(|c_j|)
    structural_pen = structural_complexity(complete_F, complete_U)

    sign_pen = 0.0
    constants_pen = 0.0
    fisher_pen = 0.0
    if settings.mode == :all_coefficients
        joint_consts = Vector{Float64}(vcat(
            get_optimizable_constants(ind.F), get_optimizable_constants(ind.U)))
        sign_pen = -0.5 * length(joint_consts) * log(3)
        constants_pen = isempty(joint_consts) ? 0.0 :
            sum(log.(max.(abs.(joint_consts), 1e-16)))
        fisher_pen = fisher_complexity(fisher_diags, fisher_status)
    end

    total = 0.0#functional_pen + structural_pen + sign_pen + constants_pen + fisher_pen
    return (
        functional=functional_pen,
        structural=structural_pen,
        sign=sign_pen,
        constants=constants_pen,
        fisher=fisher_pen,
        total=total,
    )
end


function total_penalty(ind::Individual, fisher_diags::AbstractVector{Float64}=Float64[];
    fisher_status::Symbol=:valid, settings::PenaltySettings=PenaltySettings())
    return penalty_components(ind, fisher_diags; fisher_status, settings).total
end





# ===========================================
# Evolutionary steps of the Genetic Algorithm
# ===========================================

"""
    tournament_index(scores, eligible, tour_size; select_best)

Select one index from a tournament of distinct eligible population members. Lower
description scores are better.

Generated with GPT 5.6.
"""
function tournament_index(rng::AbstractRNG, scores::Vector{Float64}, eligible::Vector{Int}, tour_size::Int;
    select_best::Bool)::Int
    isempty(eligible) && throw(ArgumentError("Tournament requires at least one eligible index."))
    1 <= tour_size <= length(eligible) || throw(ArgumentError(
        "tour_size must be between 1 and the number of eligible individuals " *
        "($(length(eligible)); got $tour_size)."))
    all(i -> 1 <= i <= length(scores), eligible) || throw(ArgumentError(
        "Eligible indices must refer to entries in scores."))

    tournament = eligible[randperm(rng, length(eligible))[1:tour_size]]
    position = select_best ? argmin(@view scores[tournament]) :
        argmax(@view scores[tournament])
    return tournament[position]
end

"""
    select_distinct_tournament_indices(scores, eligible, count, tour_size; select_best)

Select `count` distinct indices by repeatedly running tournaments. A selected
individual is removed before the next tournament, which guarantees termination.

Generated with GPT 5.6.
"""
function select_distinct_tournament_indices(rng::AbstractRNG, scores::Vector{Float64}, eligible::Vector{Int},
    count::Int, tour_size::Int; select_best::Bool)::Vector{Int}
    0 <= count <= length(eligible) || throw(ArgumentError(
        "count must be between 0 and the number of eligible individuals " *
        "($(length(eligible)); got $count)."))
    length(unique(eligible)) == length(eligible) || throw(ArgumentError(
        "Eligible indices must be distinct."))

    available_indices = copy(eligible)
    selected_indices = Int[]
    sizehint!(selected_indices, count)

    for _ in 1:count
        # Near the end, every remaining individual participates if fewer than
        # `tour_size` candidates are available.
        effective_tour_size = min(tour_size, length(available_indices))
        selected_idx = tournament_index(rng, scores, available_indices, effective_tour_size;
            select_best=select_best)
        push!(selected_indices, selected_idx)

        selected_position = findfirst(==(selected_idx), available_indices)
        deleteat!(available_indices, selected_position)
    end
    return selected_indices
end

"""
    select_replacement_indices(scores, replacement_count, tour_size)

Choose distinct parents and replacement targets. The current champion is
explicitly excluded from the loser pool, so it always survives the generation.

Generated with GPT 5.6.
"""
function select_replacement_indices(rng::AbstractRNG, scores::Vector{Float64}, replacement_count::Int,
    tour_size::Int)
    population_size = length(scores)
    2 <= replacement_count <= population_size - 1 || throw(ArgumentError(
        "replacement_count must be between 2 and Npops - 1; got " *
        "$replacement_count for Npops=$population_size."))
    iseven(replacement_count) || throw(ArgumentError("replacement_count must be even."))
    1 <= tour_size <= population_size || throw(ArgumentError(
        "tour_size must be between 1 and Npops=$population_size; got $tour_size."))

    all_indices = collect(1:population_size)
    champion = argmin(scores)

    parent_indices = select_distinct_tournament_indices(rng, scores, all_indices,
        replacement_count, tour_size; select_best=true)
    shuffle!(rng, parent_indices)

    # The champion is never eligible for replacement, which is the elitism
    # guarantee of this selection scheme.
    non_champion_indices = [i for i in all_indices if i != champion]
    loser_indices = select_distinct_tournament_indices(rng, scores, non_champion_indices,
        replacement_count, tour_size; select_best=false)

    return (parents=parent_indices, losers=loser_indices, champion=champion)
end

"""Generate children from paired tournament-selected parent indices.

Generated with GPT 5.6.
"""
function generate_children(rng::AbstractRNG, population::Vector{Individual}, parent_indices::Vector{Int}, settings,
    current_gen::Int)
    iseven(length(parent_indices)) || throw(ArgumentError("Parent count must be even."))
    all(i -> 1 <= i <= length(population), parent_indices) || throw(ArgumentError(
        "Parent indices must refer to entries in population."))
    children = Vector{Individual}(undef, length(parent_indices))

    for i in 1:2:length(parent_indices)
        parent1 = population[parent_indices[i]]
        parent2 = population[parent_indices[i + 1]]

        # Try to apply crossover
        if rand(rng) < settings.crossover_rate
            child1, child2 = crossover_sr(rng, parent1, parent2, current_gen, settings.max_size)
        else
            child1_F, child1_U = copy_node(parent1.F), copy_node(parent1.U)
            child2_F, child2_U = copy_node(parent2.F), copy_node(parent2.U)

            child1 = Individual(child1_F, child1_U, current_gen, parent1.complexity)
            child2 = Individual(child2_F, child2_U, current_gen, parent2.complexity)
        end

        # Try to apply mutation
        if rand(rng) < settings.mutation_rate
            child1 = mutate_sr(rng, child1, current_gen, settings.max_size)
        end
        if rand(rng) < settings.mutation_rate
            child2 = mutate_sr(rng, child2, current_gen, settings.max_size)
        end

        children[i] = child1
        children[i + 1] = child2
    end

    return children
end


tournament_index(scores::Vector{Float64}, eligible::Vector{Int}, tour_size::Int; select_best::Bool) =
    tournament_index(Random.default_rng(), scores, eligible, tour_size; select_best)

select_distinct_tournament_indices(scores::Vector{Float64}, eligible::Vector{Int}, count::Int,
    tour_size::Int; select_best::Bool) = select_distinct_tournament_indices(
        Random.default_rng(), scores, eligible, count, tour_size; select_best)

select_replacement_indices(scores::Vector{Float64}, replacement_count::Int, tour_size::Int) =
    select_replacement_indices(Random.default_rng(), scores, replacement_count, tour_size)

generate_children(population::Vector{Individual}, parent_indices::Vector{Int}, settings,
    current_gen::Int) = generate_children(
        Random.default_rng(), population, parent_indices, settings, current_gen)



# ==========================================
# Constant management for joint optimization
# ==========================================

# We distinguish real and integer constants for the simplification penalty log|c_j|
function get_optimizable_constants(tree::Node{Float64})
    consts = Float64[]
    for node in tree
        if node.degree == 0 && node.constant
            if !isinteger(node.val)
                push!(consts, node.val)
            end
        end
    end
    return consts
end



# # Insert vector of constants back into the tree
function set_optimizable_constants!(tree::Node{Float64}, consts::AbstractVector{Float64})
    # Ref is used to track the position in the array as we walk the tree
    _set_optimizable_constants!(tree, consts, Ref(1))
end



function _set_optimizable_constants!(node::Node{Float64}, consts::AbstractVector{Float64}, idx::Base.RefValue{Int})
    if node.degree == 0
        if node.constant && !isinteger(node.val)
            node.val = consts[idx[]]
            idx[] += 1
        end
    elseif node.degree == 1
        _set_optimizable_constants!(node.l, consts, idx)
    elseif node.degree == 2
        _set_optimizable_constants!(node.l, consts, idx)
        _set_optimizable_constants!(node.r, consts, idx)
    end
end

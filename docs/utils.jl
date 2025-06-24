using Corleone
using Base.Docs

function replace_general_headers(content::String)::String
    # This regex looks for a '#' at the beginning of a line, followed by optional spaces,
    # then captures any characters until the end of the line.
    # The 'm' flag makes '^' and '$' match start/end of lines, not just start/end of string.
    # The `s"..."` syntax is for substitution strings in Julia, where \1 refers to the first captured group.
    return replace(content, r"^#\s*(.+)$"m => s"**\1**")
end

function add_admonition(name::Symbol, content)
    ":::{.callout appearance=\"minimal\" collapse=\"true\"}\n" * "### $(name) {#doc-$(name) .unnumbered}\n" * content * "\n:::"
end

function no_docs_found(binding::Docs.Binding)
    add_admonition(binding.var, "$(binding.mod).$(binding.var)\n\n No documentation found.")
end

function process_docstring(mod::Module, name::Symbol, url=nothing)
    s = getproperty(mod, name)
    content = Docs.doc(s) |> string |> replace_general_headers
    if !isnothing(url)
        content *= "\n [Source]($(url)) \n"
    end
    add_admonition(name, content)
end

function process_docstring(io::IOStream, mod::Module, name::Symbol, url=nothing)
    write(io, process_docstring(mod, name, url))
end

autodoc(mod, outfile, baseurl=pkgdir(mod), repo="https://kosinus.math.uni-magdeburg.de/mathopt/software/corleone/-/blob/main/") = begin
    meta = Docs.meta(mod)
    lasti = length(meta)
    open(outfile, "a") do io
        for (i, (sig, data)) in enumerate(meta)
            d = map(Base.Fix1(getindex, data.docs), data.order)[1]
            url = repo * string(relpath(d.data[:path], baseurl)) * "?ref_type=heads#" * string(d.data[:linenumber])
            process_docstring(io, mod, sig.var, url)
            if i < lasti
                write(io, "\n")
            end
        end
    end

end

quarto_doc(mod::Module, var::Symbol; kwargs...) = quarto_doc(Docs.Binding(mod, var); kwargs...)

function quarto_doc(mod::Module, vars::Symbol...; kwargs...)
    docstrings = map(vars) do var
        quarto_doc(mod, var; kwargs...)
    end
    prod([i == lastindex(docstrings) ? docstrings[i] : docstrings[i] * "\n" for i in eachindex(docstrings)])
end

function quarto_doc(binding::Docs.Binding; baseurl=pkgdir(binding.mod), repo="https://kosinus.math.uni-magdeburg.de/mathopt/software/corleone/-/blob/main/", kwargs...)
    if haskey(Docs.meta(binding.mod), binding)
        d = Docs.docstr(binding)
        url = repo * string(relpath(d.data[:path], baseurl)) * "?ref_type=heads#" * string(d.data[:linenumber])
        return process_docstring(binding.mod, binding.var, url)
    end
    return no_docs_found(binding)
end


function extract_code_blocks(text::AbstractString, pattern="{julia}", exclude=default_exclude)
    # This regex captures the content between ```julia and ```
    # The `s` flag makes `.` match newlines.
    # The `(?s)` at the beginning is an inline modifier for dotall mode.
    output = ""
    for m in eachmatch(Regex("```$(pattern)\n(?s)(.*?)\n```"), text)
        if !exclude(m.captures[1])
            # Split the content into lines
            lines = split(m.captures[1], '\n')
            # Filter out lines that start with '#|' (and also trim whitespace from remaining lines)
            filtered_lines = [strip(line) for line in lines if !startswith(strip(line), "#|")]
            # Join the filtered lines back into a single string
            push!(filtered_lines, "\n")
            output *= join(filtered_lines, '\n')
        end
    end
    return output
end

function extract_test_blocks(text::AbstractString)
    output = ""
    for m in eachmatch(Regex("```julia\n(?s)(.*?)\n```"), text)
        lines = split(m.captures[1], '\n')
        filtered_lines = [line for line in lines if startswith(strip(line), "@test")]
        # Split the content into lines
        output *= join(filtered_lines, '\n')
    end
    return output
end

# Per default we exclude versioninfo or testing
function default_exclude(match::AbstractString)
    occursin("using Test", match) || occursin("versioninfo()", match)
end

# If we have test we do not want 
function default_exclude_test(match::AbstractString)
    occursin("using CairoMakie", match) || occursin("versioninfo()", match)
end

function parse_example_file(input, output; pattern="{julia}", exclude=default_exclude, kwargs...)
    content = read(input, String)
    raw_code = extract_code_blocks(content, pattern, exclude)
    write(output, raw_code)
    return true
end

function parse_example_file_to_test(input, output; pattern="{julia}", exclude=default_exclude_test, kwargs...)
    content = read(input, String)
    # Extract the raw code
    raw_code = extract_code_blocks(content, pattern, exclude)
    # Extract the tests
    test_code = extract_test_blocks(content)
    isempty(test_code) && return false
    # Append Test
    fname = first(split(basename(input), "."))
    raw_code = "using Test\n" * raw_code * test_code
    if isfile(output) && raw_code != read(output, String)
        write(output, raw_code)
        return true
    end
    return false 
end


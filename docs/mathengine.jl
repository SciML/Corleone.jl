# PlutoStaticHTML emits $$...$$ for display math and \(...\) for inline math.
# Match display delimiters before $ so equations are not split into inline math.
function notebook_mathengine()
    return Documenter.KaTeX(
        Dict(
            :delimiters => [
                Dict(:left => raw"$$", :right => raw"$$", :display => true),
                Dict(:left => raw"\[", :right => raw"\]", :display => true),
                Dict(:left => raw"\(", :right => raw"\)", :display => false),
                Dict(:left => raw"$", :right => raw"$", :display => false),
            ]
        )
    )
end

# The root gallery lets DemoCards generate this heading. The standalone manual
# uses Literate directly, so generate the same heading from the shared metadata.
function tutorial_source(content)
    frontmatter = match(r"\A# ---\r?\n((?:# .*\r?\n)*)# ---\r?\n", content)
    isnothing(frontmatter) && error("Expected native DemoCards frontmatter")
    metadata = YAML.load(replace(frontmatter.captures[1], r"^# "m => ""))
    body = replace(content, frontmatter.match => ""; count = 1)
    return "# # [$(metadata["title"])](@id $(metadata["id"]))\n" * body
end

# MaterialDocs 0.2 does not render DocumenterCitations' custom AST nodes.
# Keep the existing bibliography and citation anchors in the Material output.
module MaterialCitations

import DocumenterCitations
import MaterialDocs

escape_attribute(value) = replace(
    value, '&' => "&amp;", '"' => "&quot;", '<' => "&lt;", '>' => "&gt;"
)

function MaterialDocs.domify(
        ctx::MaterialDocs.DomifyContext, node,
        citation::DocumenterCitations.CitationSiteNode
    )
    print(ctx.io, "<span id=\"", escape_attribute(citation.id), "\">")
    MaterialDocs.domify_children(ctx, node)
    print(ctx.io, "</span>")
    return nothing
end

function MaterialDocs.domify(
        ctx::MaterialDocs.DomifyContext, node,
        bibliography::DocumenterCitations.BibliographyNode
    )
    tag = string(bibliography.list_style)
    println(ctx.io, "<", tag, " class=\"bibliography\">")
    for item in bibliography.items
        entry_tag = tag == "dl" ? "dt" : "li"
        print(ctx.io, "<", entry_tag)
        if !isnothing(item.anchor_key)
            print(ctx.io, " id=\"", escape_attribute(item.anchor_key), "\"")
        end
        print(ctx.io, ">")
        if tag == "dl"
            isnothing(item.label) || MaterialDocs.domify(ctx, item.label)
            print(ctx.io, "</dt><dd>")
        end
        MaterialDocs.domify(ctx, item.reference)
        println(ctx.io, tag == "dl" ? "</dd>" : "</li>")
    end
    println(ctx.io, "</", tag, ">")
    return nothing
end

end

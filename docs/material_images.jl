# MaterialDocs 0.2 renders Documenter's source-root image paths verbatim.
# Resolve them from each output page, as its navigation and stylesheet links do.
module MaterialImages

import MaterialDocs
const MarkdownAST = MaterialDocs.MarkdownAST

function MaterialDocs.domify(
        ctx::MaterialDocs.DomifyContext, node::MarkdownAST.Node,
        image::MaterialDocs.Documenter.LocalImage
    )
    destination = ctx.root_prefix * image.path
    alt = MaterialDocs._collect_text(node)
    escape = MaterialDocs._html_escape
    println(ctx.io, "<figure class=\"md-figure\">")
    println(ctx.io, "<img src=\"", escape(destination), "\" alt=\"", escape(alt), "\" loading=\"lazy\">")
    isempty(alt) || println(ctx.io, "<figcaption>", escape(alt), "</figcaption>")
    println(ctx.io, "</figure>")
    return nothing
end

end

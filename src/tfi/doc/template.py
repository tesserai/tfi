from string import Template

_style_fragment = """
.method-list-item {
  margin: 0;
}

.method-list {
  padding: 8px 0 0;
}

.method-list-title {
  margin: 0;
}

.method-copy .method-list-group {
  margin-top: 8px;
  border-top: 1px solid #e1e8ed;
}

.method-list-item-label {
  margin: 0;
  position: relative;
  float: left;
  text-align: right;
  width: 100px;
}

.method-list-item-description {
  margin-left: 120px;
  margin-bottom: 0;
  margin-top: 0;
}

.method-list, .method-list-item:after {
  clear: both;
}

.method-example {
  margin-left: calc(50% - 984px / 2 + 648px);
}
"""

def _render_sections(sections):
    def render_section(type, contents):
        if type == "text":
            s = """<p>%s</p>""" % "\n".join(contents)
            return s.replace("\n\n", "</p><p>")

        if type == "example":
            return "\n".join(contents)

    return "\n".join([
        render_section(type, contents)
        for type, contents in sections
    ])

_page_template = """
<!doctype html>
<meta charset="utf-8">
<script src="https://distill.pub/template.v1.js"></script>

<script type="text/front-matter">
  title: $title
  description: $subhead
  authors:
$authors_fragment
  affiliations:
$affiliations_fragment
</script>


<style>
$style_fragment
</style>

<dt-article>
<h1>$title</h1>
<h2>$subhead</h2>

<dt-byline></dt-byline>

$sections_fragment

<h2>Methods</h2>
$methods_fragment

</div>
</dt-article>

<dt-appendix>
</dt-appendix>

<script type="text/bibliography">
$bibliography_fragment
</script>
"""

def write(path, **kwargs):
    with open(path, "w") as f:
        f.write(render(**kwargs))

def render(
        authors,
        title,
        subhead,
        sections,
        methods,
        bibliographies):
    t = Template(_page_template)
    return t.substitute(
            authors_fragment="\n".join([
                "  - %s: %s" % (author["name"], author["url"])
                for author in authors
            ]),
            affiliations_fragment="\n".join([
                "  - %s: %s" % (author["affiliation_name"], author["affiliation_url"])
                for author in authors
            ]),
            style_fragment=_style_fragment,
            title=title,
            subhead=subhead,
            sections_fragment=_render_sections(sections),
            methods_fragment="\n".join([
                Template("""<div class="method-area">
                  <div class="method-copy" style="font-size: 1rem">
                    <dt-code class="language-tensorlang">$signature</dt-code>

                    $sections_fragment

                    <div class="method-list inputs">
                      <h4 class="method-list-title">ARGS</h4>
                      <div class="method-list-group">
                        $args_fragment
                      </div>
                    </div>

                    <div class="method-list outputs">
                      <h4 class="method-list-title">RETURNS</h4>
                      <div class="method-list-group">
                        $returns_fragment
                      </div>
                    </div>
                  </div>""").substitute(
                        args_fragment="\n".join([
                                Template("""
                                        <div class="method-list-item">
                                          <h5 class="method-list-item-label">$name</h5>
                                          <p class="method-list-item-description">$description</p>
                                        </div>""").substitute(
                                        name=name,
                                        description="\n".join(description))
                                for name, kind, description in method['args']]),
                        returns_fragment="\n".join([
                                Template("""
                                        <div class="method-list-item">
                                          <h5 class="method-list-item-label">$name</h5>
                                          <p class="method-list-item-description">$description</p>
                                        </div>""").substitute(
                                        name=name,
                                        description="\n".join(description))
                                for name, kind, description in method['returns']]),
                        sections_fragment=_render_sections(method['sections']),
                        **method)
                for method in methods
            ]),
            bibliography_fragment="\n".join(bibliographies))
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

def _render_section(type, contents):
    if type == "text":
        s = """<p>%s</p>""" % "\n".join(contents)
        return s.replace("\n\n", "</p><p>")

    if type == "example":
        return "\n".join(contents)


_page_template = """
<html>
  <head>
    <meta content="IE=edge" http-equiv="X-UA-Compatible">
    <meta charset="utf-8">
    <meta content="width=device-width,initial-scale=1.0,minimum-scale=1.0,maximum-scale=1.0,user-scalable=no" name="viewport">

    <style>
    $style_fragment
    </style>
  </head>

<!--
<script type="text/front-matter">
  title: $title
  description: $subhead
  authors:
$authors_fragment
  affiliations:
$affiliations_fragment
</script>
-->

  <body id="documentation" class="">
    <div id="content">
    <h1>$title</h1>
    <h2>$subhead</h2>

    <dt-byline></dt-byline>

    $sections_fragment
    $methods_fragment
    </div>

  </body>
</html>

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
        references):
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
            sections_fragment="\n".join([
                    _render_section(type, contents)
                for type, contents in sections]),
            methods_fragment="\n".join([
                Template("""<section class="method">
                  <div class="method-area">
                    <div class="method-copy">
                      <div class="method-copy-padding">
                        <h1>$method_name method</h1>
                        $sections_fragment
                      </div>
                      <div class="method-list inputs">
                        <h4 class="method-list-title">INPUTS</h4>
                        <div class="method-list-group">
                          $args_fragment
                        </div>
                      </div>
                      <div class="method-list outputs">
                        <h4 class="method-list-title">OUTPUTS</h4>
                        <div class="method-list-group">
                          $returns_fragment
                        </div>
                      </div>
                      </div>
                    </div>
                  </div>
                </section>""").substitute(
                        method_name=method['name'],
                        args_fragment="\n".join([
                                Template("""
                                        <div class="method-list-item">
                                          <h5 class="method-list-item-label">
                                            $name
                                            <span class="method-list-item-label-details">$details</span>
                                          </h5>
                                          <p class="method-list-item-description">$description</p>
                                        </div>""").substitute(
                                        name=name,
                                        details=kind,
                                        description="\n".join(description))
                                for name, kind, description in method['args']]),
                        returns_fragment="\n".join([
                                Template("""
                                        <div class="method-list-item">
                                          <h5 class="method-list-item-label">
                                            $name
                                            <span class="method-list-item-label-details">$details</span>
                                          </h5>
                                          <p class="method-list-item-description">$description</p>
                                        </div>""").substitute(
                                        name=name,
                                        details=kind,
                                        description="\n".join(description))
                                for name, kind, description in method['returns']]),
                        sections_fragment="\n".join([
                                _render_section(type, contents)
                                for type, contents in method['sections']]),
                        **method)
                for method in methods
            ]),
            bibliography_fragment="\n".join(references))

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

from string import Template

_style_fragment = HtmlFormatter(style='paraiso-dark').get_style_defs('.language-python')

with open(__file__[:-2] + "css") as f:
    _style_fragment += f.read()

# ['default', 'emacs', 'friendly', 'colorful', 'autumn', 'murphy', 'manni', 'monokai', 'perldoc', 'pastie', 'borland', 'trac', 'native', 'fruity', 'bw',
# 'vim', 'vs', 'tango', 'rrt', 'xcode', 'igor', 'paraiso-light', 'paraiso-dark', 'lovelace', 'algol', 'algol_nu', 'arduino', 'rainbow_dash', 'abap']

from pygments.styles import get_all_styles
print(list(get_all_styles()))

def _render_section(type, contents):
    if type == "text":
        s = """<p>%s</p>""" % "\n".join(contents)
        return s.replace("\n\n", "</p><p>")

    if type == "example":
        return "\n".join(contents)

    return ""


_page_template = """<!doctype html>
<html>
  <head>
    <meta content="IE=edge" http-equiv="X-UA-Compatible">
    <meta charset="utf-8">
    <meta content="width=device-width,initial-scale=1.0,minimum-scale=1.0,maximum-scale=1.0,user-scalable=no" name="viewport">

    <link rel="stylesheet" href="../design/fonts/whitney.css">

    <style>
    $style_fragment
    </style>
  </head>

  <body id="documentation" class="">
    <!-- <img src="../design/steps/TFI_v2/preview/page-1-desktop-hd-copy-20.png" style="position: absolute; overflow: hidden; opacity: 0.5; z-index: -1; left: 0; top: 0; width: 1440px;"/> -->
    <div id="content">
    <div id="head">
        <div class="source-code-ref">
          <a href="$source_repo_url">$source_repo_label</a>
          <div class="commit">commit $source_repo_commit</div>
        </div>

        <h1>$title</h1>
        <h2>$subhead</h2>

        <div id="authorship">
        $authors_fragment
        </div>
    </div>
    <div id="rest">
        <div id="background"><div class="background-actual"></div></div>
        $sections_fragment
        $methods_fragment

        <div id="appendix">
            <section class="hyperparameters">
                <h3>Hyperparameters</h3>
                <div class="method-area">
                    <div class="method-copy">
                        <div class="method-copy-padding">
                            $hyperparameters_fragment
                        </div>
                    </div>
                </div>
            </section>
            <section class="implementation-notes">
                <h3>Implementation Notes</h3>
                $implementation_notes_fragment
            </section>
            <section class="references">
                <h3>References</h3>
                <ol>
                    $references_fragment
                </ol>
            </section>
        </div>
    </div>
  </body>
</html>

<script type="text/bibliography">

  $bibliography_fragment
</script>
"""




# // From https://github.com/distillpub/template/blob/master/components/citation.js
# // TODO(adamb) Port this parsing logic to Python and use it to generate links.
#   function link_string(ent){
#     if ("url" in ent){
#       var url = ent.url;
#       var arxiv_match = (/arxiv\.org\/abs\/([0-9\.]*)/).exec(url);
#       if (arxiv_match != null){
#         url = `http://arxiv.org/pdf/$${arxiv_match[1]}.pdf`;
#       }
#
#       if (url.slice(-4) == ".pdf"){
#         var label = "PDF";
#       } else if (url.slice(-5) == ".html") {
#         var label = "HTML";
#       }
#       return ` &ensp;<a href="$${url}">[$${label||"link"}]</a>`;
#     } else {
#       return "";
#     }
#   }
#
#   function venue_string(ent) {
#     var cite = (ent.journal || ent.booktitle || "")
#     if ("volume" in ent){
#       var issue = ent.issue || ent.number;
#       issue = (issue != undefined)? "("+issue+")" : "";
#       cite += ", Vol " + ent.volume + issue;
#     }
#     if ("pages" in ent){
#       cite += ", pp. " + ent.pages
#     }
#     if (cite != "") cite += ". "
#     if ("publisher" in ent){
#       cite += ent.publisher;
#       if (cite[cite.length-1] != ".") cite += ".";
#     }
#     return cite;
#   }
#
#
#   function doi_string(ent, new_line){
#     if ("doi" in ent) {
#       return `$${new_line?"<br>":""} <a href="https://doi.org/$${ent.doi}" style="text-decoration:inherit;">DOI: $${ent.doi}</a>`;
#     } else {
#       return "";
#     }
#   }
#
#   function bibliography_cite(ent, fancy){
#     if (ent){
#       var cite =  "<b>" + ent.title + "</b> "
#       cite += link_string(ent) + "<br>";
#       cite += author_string(ent, "$${L}, $${I}", ", ", " and ");
#       if (ent.year || ent.date){
#         cite += ", " + (ent.year || ent.date) + ". "
#       } else {
#         cite += ". "
#       }
#       cite += venue_string(ent);
#       cite += doi_string(ent);
#       return cite
#     } else {
#       return "?";
#     }
#   }







def write(path, **kwargs):
    with open(path, "w") as f:
        f.write(render(**kwargs))

from pprint import pprint

def render(
        source,
        authors,
        title,
        subhead,
        sections,
        methods,
        references):
    # pprint(methods)

    t = Template(_page_template)

    hyperparameters = [
        ("inputs", "value is placeholder float32 &lt;?, 256, 256, 3&gt;", ["a tensor of size [batch_size, height, width, channel]"]),
        ("num_classes", "float32 &lt;32&gt;", ["number of predicted classes"]),
        ("inputs", "value is placeholder float32 &lt;?, 256, 256, 3&gt;", ["a tensor of size [batch_size, height, width, channel]"]),
        ("num_classes", "float32 &lt;32&gt;", ["number of predicted classes"]),
        ("inputs", "value is placeholder float32 &lt;?, 256, 256, 3&gt;", ["a tensor of size [batch_size, height, width, channel]"]),
        ("num_classes", "float32 &lt;32&gt;", ["number of predicted classes"]),
        ("inputs", "value is placeholder float32 &lt;?, 256, 256, 3&gt;", ["a tensor of size [batch_size, height, width, channel]"]),
        ("num_classes", "float32 &lt;32&gt;", ["number of predicted classes"]),
    ]


    implementation_notes = [
        [
            """Training for image classification on Imagenet is usually done with [224, 224] inputs, resulting in [7, 7] feature maps at the output of the last ResNet block for the ResNets defined in [1] that have nominal stride equal to 32. However, for dense prediction tasks we advise that one uses inputs with spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In this case the feature maps at the ResNet output will have spatial shape [(height - 1) / output_stride + 1,""",
            """(width - 1) / output_stride + 1] and corners exactly aligned with the input image corners, which greatly facilitates alignment of the features to the image. Using as input [225, 225] images results in [8, 8] feature maps at the output of the last ResNet block.""",
        ],
        ["""For dense prediction tasks, the ResNet needs to run in fully-convolutional (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all have nominal stride equal to 32 and a good choice in FCN mode is to use output_stride=16 in order to increase the density of the computed features at small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915."""]
    ]

    def render_arg(**kw):
        return Template("""
                <li class="method-list-item">
                  <h5 class="method-list-item-label">
                    $name
                    <span class="method-list-item-label-details">$details</span>
                  </h5>
                  <p class="method-list-item-description">$description</p>
                </li>""").substitute(**kw)

    def render_args(klass, args, example_klass, example, expanded=[]):
        example_fragment = ""

        if example is not None:
            highlighted = highlight("\n".join(example),
                                    PythonLexer(),
                                    HtmlFormatter(cssclass='language-python'))
            example_fragment += """
                <div class="method-example">
                    <div class="method-example-part">
                        <div class="method-example-%s">
                            %s
                        </div>
                    </div>
                </div>
                """ % (example_klass, highlighted)

        for name, e in expanded:
            example_fragment += """
            <div class="method-example">
                <div class="method-example-part">
                    <div class="method-example-%s">
                        %s%s
                    </div>
                </div>
            </div>
            """ % ('expanded', name, e)

        if len(args) == 0:
            return example_fragment

        return Template("""
          <div class="method-area">
            <div class="method-copy">
              <div class="method-copy-padding">
                <div class="method-list $klass">
                  <h4 class="method-list-title"></h4>
                  <ul class="method-list-group">
                    $args_fragment
                  </ul>
                </div>
              </div>
            </div>
            $example_fragment
          </div>
                """).substitute(
                klass=klass,
                args_fragment="\n".join([
                        render_arg(name=name,
                            details=kind.replace("<", "&lt;").replace(">", "&gt;"),
                            description="\n".join(description))
                        for name, kind, description in args]),
                example_fragment=example_fragment)

    return t.substitute(
            source_repo_url=source['url'],
            source_repo_label=source['label'],
            source_repo_commit=source['commit'],
            authors_fragment="\n".join([
                Template("""<span class="author">
                  <span class="author-name"><a href="$url">$author</a></span>
                  <span class="author-role"><a href="$role_url">$role</a></span>
                </span>""").substitute(
                    author=author['name'],
                    url=author['url'],
                    role=author['affiliation_name'],
                    role_url=author['affiliation_url'])
                for author in authors]),
            style_fragment=_style_fragment,
            title=title,
            subhead=subhead,
            sections_fragment="\n".join([
                _render_section(type, contents)
                for type, contents in sections]),
            methods_fragment="\n".join([
                Template("""
            <section class="method">
                <div class="method-area">
                    <div class="method-copy">
                        <div class="method-copy-padding">
                            <h1>$method_name method</h1>
                            $sections_fragment
                        </div>
                    </div>
                    <div class="method-example">
                        <div class="method-example-part">
                            <div class="method-example-declaration">
                                <pre class="language-python"><code class="language-python">foo</code></pre>
                            </div>
                        </div>
                    </div>
                </div>
                $args_fragment
                $returns_fragment
            </section>
                """).substitute(
                        method_name=method['name'],
                        args_fragment=render_args('inputs', method['args'], 'input', method['example args'], method['example expanded']),
                        returns_fragment=render_args('outputs', method['returns'], 'output', None, [('result', method['example result'])]),
                        sections_fragment="\n".join([
                                _render_section(type, contents)
                                for type, contents in method['sections']]),
                        **method)
                for method in methods
            ]),
            implementation_notes_fragment="\n".join([
                    "<p>%s</p>" % "\n".join(implementation_notes_p)
                    for implementation_notes_p in implementation_notes]),
            hyperparameters_fragment=render_args('hyperparameters', hyperparameters, None, None),
            bibliography_fragment=references,
            references_fragment="\n".join([
                Template("""<li>
                    <b>$title</b>&nbsp;&nbsp;&nbsp;<a href="$pdf_url">[PDF]</a>
                    <br>
                    $authors_fragment
                </li>""").substitute(
                    title=title,
                    pdf_url=pdf_url,
                    authors_fragment=", ".join(authors)
                )
                for title, pdf_url, authors in references]))

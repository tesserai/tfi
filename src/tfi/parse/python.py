def parse_example_args(example_args, locals_dict):
    # TODO(adamb) Confirm we can properly parse k as an id and v alone.
    python_kw_src = ", ".join([
        "%s=%s" % (name, "\n".join(doc))
        for name, type, doc in example_args
    ])
    args_src = """
import tfi.data
_ = dict(%s)
""" % python_kw_src
    g = {}
    exec(args_src, g, locals_dict)
    return locals_dict['_']

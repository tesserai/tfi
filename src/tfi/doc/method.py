class MethodDocumentation(object):
    def __init__(self, *, name, overview, inputs, outputs, examples):
        self._name = name # str
        self._overview = overview # [str, ...]
        self._inputs = inputs # [(name, tensor_info_str, doc: [str])]
        self._outputs = outputs # [(name, tensor_info_str, doc: [str])]
        self._examples = examples

    def name(self): return self._name

    def overview(self): return self._overview

    def inputs(self): return self._inputs

    def outputs(self): return self._outputs

    def example(self): return self.examples()[0]

    def examples(self): return self._examples

    def with_updated_example_outputs(self, method):
        return MethodDocumentation(
            name=self._name,
            overview=self._overview,
            inputs=self._inputs,
            outputs=self._outputs,
            examples=[
              example.with_updated_outputs(method)
              for example in self._examples
            ],
        )

    def docstring(self):
        name = self._name
        overview = self._overview
        docstr = None
        if overview:
            docstr = "\n".join([name, "-" * len(name), "", overview])

        return docstr


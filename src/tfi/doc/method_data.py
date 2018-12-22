class MethodDataDocumentation(object):
    @classmethod
    def generate(cls, *, method, inputs):
        return cls(
            inputs=inputs,
            outputs=method(**inputs)
        )

    def __init__(self, *, inputs, outputs=None):
        self._inputs = inputs
        self._outputs = outputs

    def inputs(self): return self._inputs

    def outputs(self): return self._outputs

    def with_updated_outputs(self, method):
        return MethodDataDocumentation.generate(
            method=method,
            inputs=self._inputs,
        )


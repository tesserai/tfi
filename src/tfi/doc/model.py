class ModelDocumentation(object):
    def __init__(self, *,
        hyperparameters,
        name,
        overview,
        methods,
        authors,
        references,
        implementation_notes,
        source,
        facets_overview_proto,
    ):
        self._hyperparameters = hyperparameters
        self._name = name
        self._overview = overview
        self._methods = methods
        self._authors = authors
        self._references = references
        self._implementation_notes = implementation_notes
        self._source = source
        self._facets_overview_proto = facets_overview_proto

    def hyperparameters(self): return self._hyperparameters

    def name(self): return self._name

    def overview(self): return self._overview

    def methods(self): return self._methods

    def authors(self): return self._authors

    def references(self): return self._references

    def implementation_notes(self): return self._implementation_notes

    def source(self): return self._source

    def facets_overview_proto(self): return self._facets_overview_proto

    def with_updated_example_outputs(self, model):
        return ModelDocumentation(
            hyperparameters=self._hyperparameters,
            name=self._name,
            overview=self._overview,
            authors=self._authors,
            references=self._references,
            implementation_notes=self._implementation_notes,
            source=self._source,
            facets_overview_proto=self._facets_overview_proto,
            methods=[
                method.with_updated_example_outputs(getattr(model, method.name()))
                for method in self._methods
            ],
        )

    def docstring(self):
        name = self._name
        overview = self._overview
        docstr = None
        if overview:
            docstr = "\n".join([name, "-" * len(name), "", overview])

        return docstr

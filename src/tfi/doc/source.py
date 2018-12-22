from tfi.resolve.git import git_authorship as _git_authorship
from tfi.resolve.git import GitUserRepo as _GitUserRepo

class ModelSource(object):
    @classmethod
    def detect(cls, model):
        git_authorship_file = None
        if hasattr(model, '__file__'):
            git_authorship_file = model.__file__
        elif hasattr(model, '__tfi_file__'):
            git_authorship_file = model.__tfi_file__
        elif hasattr(model, '__tfi_module__'):
            git_authorship_file = model.__tfi_module__.__file__

        if not git_authorship_file:
            return None
        github_user_repo = _GitUserRepo("github-users.json")
        git = _git_authorship(github_user_repo, git_authorship_file)
        return cls(
            url=git['url'],
            label=git['label'],
            commit=git['commit'][:7],
            authors=git['authors'],
        )

    def __init__(self, *, url, label, commit, authors):
        self._url = url
        self._label = label
        self._commit = commit
        self._authors = authors

    def url(self): self._url

    def label(self): self._label

    def commit(self): self._commit

    def authors(self): self._authors

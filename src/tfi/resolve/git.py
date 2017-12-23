import itertools
import json
import os
import subprocess

from functools import partial
from urllib.request import urlopen
from collections import OrderedDict

import tinydb
from tinydb.storages import JSONStorage
from tinydb.middlewares import CachingMiddleware

class GitUserRepo(object):
    def __init__(self, path):
        storage = CachingMiddleware(JSONStorage)
        self._db = tinydb.TinyDB(path, storage=storage)
        self._flush = storage.flush

    def close(self):
        self._db.close()

    def resolve(self, email, resolve):
        q = tinydb.Query()

        found = self._db.search(q.email == email)
        if len(found) > 0:
            return found[0]["user"]

        user = resolve(email)
        if user is None:
            return None

        print("resolved %s -> %s" % (email, user))

        items = list(user.items())
        items.sort()
        record = OrderedDict([("email", email), ("user", OrderedDict(items))])
        self._db.insert(record)
        self._flush()

        return user

_GIT_EDIT_KINDS = ['insertions', 'files', 'deletions', 'commits']

_REMOTE_GIT_REPO_CTORS = []
def find_remote_git_repo(url):
    for url_prefix, fn in _REMOTE_GIT_REPO_CTORS:
        if url.startswith(url_prefix):
            return fn(url)
    return None

def _remote_git_repo_impl(url_prefixes):
    def _wrap(fn):
        for url_prefix in url_prefixes:
            _REMOTE_GIT_REPO_CTORS.append((url_prefix, fn))
        return fn
    return _wrap

@_remote_git_repo_impl(["https://github.com/"])
class GitHubRepo(object):
    def __init__(self, url):
        self._url = url
        if url.endswith(".git"):
            url = url[:-4]
        self._base_url = url
        split = url.split("/")
        self._owner = split[-2]
        self._repo = split[-1]
        self._github_hidden_email_domain = "@users.noreply.github.com"

    def _do_resolve_user(self, email=None, recent_commit_fn=None):
        if email is not None:
            if email.endswith(self._github_hidden_email_domain):
                user = email[:-len(self._github_hidden_email_domain)]
                with urlopen("https://api.github.com/users/%s" % user) as f:
                    return json.loads(f.read())

            with urlopen("https://api.github.com/search/users?q=%s+in:email" % email) as f:
                matches = json.loads(f.read())['items']
                if len(matches) > 0:
                    return matches[0]

        if recent_commit_fn is not None:
            commit = recent_commit_fn()
            url = "https://api.github.com/repos/%s/%s/commits/%s" % (self._owner, self._repo, commit)
            with urlopen(url) as f:
                return json.loads(f.read())["author"]

    def _resolve_user(self, email_db, email, recent_commit_fn):
        resolve = partial(self._do_resolve_user, recent_commit_fn=recent_commit_fn)
        return email_db.resolve(email, resolve)

    def user_url(self, email_db, email, recent_commit_fn):
        user = self._resolve_user(email_db, email, recent_commit_fn)
        if user is not None:
            return user['html_url']

        print("Couldn't resolve user for", email)
        return "mailto:%s" % email

    def authorship_url(self, commit, subpath, email_db, email, recent_commit_fn):
        user = self._resolve_user(email_db, email, recent_commit_fn)
        if user is not None:
            author = user['login']
        else:
            author = email
        return "%s/commits/%s/%s?author=%s" % (self._base_url, commit, subpath, author)

    def tree_url(self, commit, subpath):
        return "%s/tree/%s/%s" % (self._base_url, commit, subpath)

    def label(self, subpath):
        label = self._base_url
        if label.startswith("https://"):
            label = label[8:]
        if subpath:
            label = "%s/%s" % (label, subpath)
        return label

class GitRepo(object):
    @staticmethod
    def discover(path, subpath=None, markerfile=".git"):
        if os.path.isdir(path) and os.path.exists(os.path.join(path, markerfile)):
            return GitRepo(path), subpath
        dirname, basename = os.path.split(path)
        if dirname == path:
            return None, subpath
        subpath = os.path.join(basename, subpath) if subpath else basename
        return GitRepo.discover(dirname, subpath=subpath)

    def __init__(self, path):
        self._git_repo_path = path

    def history_authorship(self, subpath):
        history_cmd = ["git", "-C", self._git_repo_path,
            "log", '--pretty=%aE%n%aN', "--shortstat",
            "--reverse",
            "--no-merges", "-M20", "-C20", "--ignore-space-change",
            '--',
            subpath,
        ]
        def each_n(iterable, n):
            it = iter(iterable)
            while True:
                chunk_it = itertools.islice(it, n)
                try:
                    first_el = next(chunk_it)
                except StopIteration:
                    return
                yield itertools.chain((first_el,), chunk_it)
        names_by_email = {}
        edits_by_email = {}
        with subprocess.Popen(history_cmd, stdout=subprocess.PIPE) as p:
            for email, name, _, edits_raw in each_n(p.stdout.readlines(), 4):
                email = email.decode().rstrip()
                name = name.decode().rstrip()
                # TODO(adamb) Include github user so we can make pretty links...
                edits = {kind[0]: int(n) for n, kind, *_ in [desc.strip().split(' ') for desc in edits_raw.decode().split(", ")]}
                edits = {
                    'files': edits.get('f', 0),
                    'insertions': edits.get('i', 0),
                    'deletions': edits.get('d', 0),
                    'commits': 1
                }
                names_by_email[email] = name
                if email not in edits_by_email:
                    edits_by_email[email] = {kind: 0 for kind in _GIT_EDIT_KINDS}
                totals = edits_by_email[email]
                edits_by_email[email] = {kind: totals[kind] + edits[kind] for kind in _GIT_EDIT_KINDS}
        return names_by_email, edits_by_email

    def blame_authorship(self, subpath):
        blame_by_email = {}
        blame_cmd = """git ls-files %s | sh -c 'while read f; do git blame --line-porcelain -M -C -w $f | grep -E "^author-mail "; done' | sort | uniq -ic | sort -rn"""
        with subprocess.Popen(blame_cmd % subpath, shell=True, cwd=self._git_repo_path, stdout=subprocess.PIPE) as p:
            for line in p.stdout.readlines():
                n, _, email = line.decode().strip().split(' ', 2)
                email = email[1:-1]
                blame_by_email[email] = int(n)
        return blame_by_email

    def commit(self, git_log_options=[]):
        with subprocess.Popen(["git", "-C", self._git_repo_path, "log", "-n", "1", "--format=%H", *git_log_options], stdout=subprocess.PIPE) as p:
            return p.stdout.read().decode().strip()

    def commit_ancestor_with_author(self, email):
        return self.commit(git_log_options=["--author=%s" % email])

    def remote_urls_with_commit(self, commit):
        remotes = set()
        with subprocess.Popen(["git", "-C", self._git_repo_path, "branch", "-r", "--contains", commit], stdout=subprocess.PIPE) as p:
            for line in p.stdout.readlines():
                remotes.add(line.decode().strip().split("/", 1)[0])

        remote_urls = set()
        for remote in remotes:
            with subprocess.Popen(["git", "-C", self._git_repo_path, "remote", "get-url", remote], stdout=subprocess.PIPE) as p:
                remote_urls.add(p.stdout.read().decode().strip())
        return remote_urls

    def remote_repos_with_commit(self, commit):
        remote_urls = self.remote_urls_with_commit(commit)
        return [find_remote_git_repo(url) for url in remote_urls]

def git_authorship(github_user_db, path):
    git_repo, subpath = GitRepo.discover(path)
    if git_repo is None:
        return None
    commit = git_repo.commit()
    repos = git_repo.remote_repos_with_commit(commit)
    if len(repos) == 0:
        return None

    remote_repo = repos[0]

    names_by_email, edits_by_email = git_repo.history_authorship(subpath)
    blame_by_email = git_repo.blame_authorship(subpath)
    contributions = OrderedDict()

    for email, edits in edits_by_email.items():
        recent_commit_fn = partial(git_repo.commit_ancestor_with_author, email)
        url = remote_repo.user_url(
                github_user_db,
                email,
                recent_commit_fn)
        if url in contributions:
            contribution = contributions[url]
            for kind in _GIT_EDIT_KINDS:
                contribution[kind] += edits[kind]
        else:
            contributions[url] = {
                'email': email,
                'url': url,
                'name': names_by_email[email],
                'commits_url': remote_repo.authorship_url(
                        commit,
                        subpath,
                        github_user_db,
                        email,
                        recent_commit_fn),
                'blames': blame_by_email.get(email, 0),
                **{kind: edits[kind] for kind in _GIT_EDIT_KINDS}
            }

    contribution_list = list(contributions.values())
    contribution_list.sort(reverse=True, key=lambda c: (c['blames'], c['insertions'], c['files'], c['deletions']))

    return {
        "commit": commit,
        "url": remote_repo.tree_url(commit, subpath),
        "label": remote_repo.label(subpath),
        "authors": contribution_list,
    }

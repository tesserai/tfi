import itertools
import os
import subprocess

def git_repo_for(path, subpath=None, markerfile=".git"):
    if os.path.isdir(path) and os.path.exists(os.path.join(path, markerfile)):
        return path, subpath
    dirname, basename = os.path.split(path)
    if dirname == path:
        return None, subpath
    subpath = os.path.join(basename, subpath) if subpath else basename
    return git_repo_for(dirname, subpath=subpath)

git_edit_kinds = ['insertions', 'files', 'deletions', 'commits']
def git_history_authorship(git_repo, subpath):
    history_cmd = ["git", "-C", git_repo,
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
                edits_by_email[email] = {kind: 0 for kind in git_edit_kinds}
            totals = edits_by_email[email]
            edits_by_email[email] = {kind: totals[kind] + edits[kind] for kind in git_edit_kinds}
    return names_by_email, edits_by_email

def git_blame_authorship(git_repo, subpath):
    blame_by_email = {}
    blame_cmd = """git ls-files %s | sh -c 'while read f; do git blame --line-porcelain -M -C -w $f | grep -E "^author-mail "; done' | sort | uniq -ic | sort -rn"""
    with subprocess.Popen(blame_cmd % subpath, shell=True, cwd=git_repo, stdout=subprocess.PIPE) as p:
        for line in p.stdout.readlines():
            n, _, email = line.decode().strip().split(' ', 2)
            email = email[1:-1]
            blame_by_email[email] = int(n)
    return blame_by_email

def git_commit(git_repo):
    with subprocess.Popen(["git", "-C", git_repo, "log", "-n", "1", "--format=%H"], stdout=subprocess.PIPE) as p:
        return p.stdout.read().decode().strip()

def git_remote_urls_with_commit(git_repo, commit):
    remotes = set()
    with subprocess.Popen(["git", "-C", git_repo, "branch", "-r", "--contains", commit], stdout=subprocess.PIPE) as p:
        for line in p.stdout.readlines():
            remotes.add(line.decode().strip().split("/", 1)[0])

    remote_urls = set()
    for remote in remotes:
        with subprocess.Popen(["git", "-C", git_repo, "remote", "get-url", remote], stdout=subprocess.PIPE) as p:
            remote_urls.add(p.stdout.read().decode().strip())
    return remote_urls

def git_authorship(path):
    git_repo, subpath = git_repo_for(path)
    commit = git_commit(git_repo)
    urls = git_remote_urls_with_commit(git_repo, commit)
    url = list(urls)[0]

    names_by_email, edits_by_email = git_history_authorship(git_repo, subpath)
    blame_by_email = git_blame_authorship(git_repo, subpath)
    contributions = []
    for email, edits in edits_by_email.items():
        contributions.append({
            'email': email,
            'name': names_by_email[email],
            'commits_url': "%s/commits/%s/%s?author=%s" % (url, commit, subpath, email),
            'blames': blame_by_email.get(email, 0),
            **{kind: edits[kind] for kind in git_edit_kinds}
        })
    contributions.sort(reverse=True, key=lambda c: (c['blames'], c['insertions'], c['files'], c['deletions']))

    return {
        "commit": commit,
        "url": url,
        "authors": contributions,
    }

import os
import pywatchman
import threading
import time
import traceback
import sys

from os.path import join, abspath
from traceback import extract_tb, format_list, format_exception_only
def _shadow(*hide):
    ''' Return a function to be set as new sys.excepthook.
        It will HIDE traceback entries for files from these directories. '''
    hide = tuple(join(abspath(p), '') for p in hide)
    def _check_file(name):
        return name and not name.startswith(hide)
    def _print(type, value, tb):
        show = (fs for fs in extract_tb(tb) if _check_file(fs.filename))
        fmt = format_list(show) + format_exception_only(type, value)
        print(''.join(fmt), end='', file=sys.stderr)
    return _print
_excepthook = _shadow(*sys.path)

class AutoRefresher(object):
    def __init__(self):
        self.client = None
        self.lock = threading.RLock()
        self.subscriptions = {} # subscription_name -> [basename, sha, refresh_fns]
        self.towatch = []

    def watch(self, source_path, initial_sha1hex, refresh_fn):
        with self.lock:
            if not self.client:
                self.towatch.append((source_path, initial_sha1hex, refresh_fn))
            else:
                self._watch(source_path, initial_sha1hex, refresh_fn)

    def _watch(self, source_path, initial_sha1hex, refresh_fn):
        subscription_name = source_path

        if subscription_name in self.subscriptions:
            self.subscriptions[subscription_name][2].append(refresh_fn)
            return

        parentdir, basename = os.path.split(source_path)
        self.client.query('subscribe', parentdir, subscription_name, {
            "expression": ["allof", ["match", basename]],
            "fields": ["name", "size", "mtime_ms", "exists", "type", "content.sha1hex"],
        })

        self.subscriptions[subscription_name] = [basename, initial_sha1hex, [refresh_fn]]

    def _run(self):
        with pywatchman.client() as c:
            with self.lock:
                self.client = c
                c.setTimeout(1)
                for entry in self.towatch:
                    self._watch(*entry)
                self.towatch.clear()

            while True:
                try:
                    # Wait for subscription events
                    with self.lock:
                        c.receive()
                    # print("receive")
                    for subscription_name, entry in self.subscriptions.items():
                        basename, last_sha, refresh_fns = entry
                        events = c.getSubscription(subscription_name)
                        # print("events", events)
                        if not events:
                            continue
                        for e in events:
                            for f in e['files']:
                                if f['name'] != basename:
                                    # print("f['name'] != basename", f['name'], "!=", basename)
                                    continue

                                sha = f['content.sha1hex']
                                if last_sha == sha:
                                    # print("last_sha == sha", last_sha, "==", sha)
                                    continue

                                # print("last_sha != sha", last_sha, "!=", sha)
                                try:
                                    for refresh_fn in refresh_fns:
                                        refresh_fn()
                                    entry[1] = sha
                                except:
                                    _excepthook(*sys.exc_info())

                except pywatchman.SocketTimeout:
                    time.sleep(.1)

    def start(self):
        threading.Thread(
            target=self._run,
            daemon=True,
        ).start()

"""Hard offline guard: any network attempt inside ``forbid_network()`` raises immediately.

MOABB's ``data_path`` silently downloads missing data, so a custom env var is not a reliable offline
guarantee. ``forbid_network`` monkeypatches the low-level network entry points (socket connect,
urllib, requests) so a missing-data download attempt becomes an instant OfflineDownloadError instead
of a slow timeout or an actual fetch. It also counts attempts for the preflight report.
"""
from __future__ import annotations

import os
import socket
import urllib.request
import warnings
from contextlib import contextmanager

from .registry import OfflineDownloadError

_MNE_ENV = ("MNE_DATA", "MNE_DATASETS_BNCI_PATH", "MNE_DATASETS_GIGADB_PATH",
            "MNE_DATASETS_LEE2019_MI_PATH", "MOABB_OFFLINE")


@contextmanager
def moabb_offline_root(datalake_root):
    """Point MNE/MOABB at the read-only datalake for the duration of a load, then restore the env and
    the global warning filters exactly. Never writes an MNE user config and never changes HOME."""
    root = str(datalake_root)
    saved = {k: os.environ.get(k) for k in _MNE_ENV}
    os.environ["MNE_DATA"] = root
    os.environ["MNE_DATASETS_BNCI_PATH"] = root
    os.environ["MNE_DATASETS_GIGADB_PATH"] = root
    os.environ["MNE_DATASETS_LEE2019_MI_PATH"] = root
    os.environ["MOABB_OFFLINE"] = "1"
    with warnings.catch_warnings():                 # restore global warning filters on exit
        warnings.simplefilter("ignore")
        try:
            yield root
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


class _Counter:
    def __init__(self):
        self.attempts = 0


@contextmanager
def forbid_network(counter: _Counter | None = None):
    counter = counter if counter is not None else _Counter()

    def blocked(*_a, **_k):
        counter.attempts += 1
        raise OfflineDownloadError("network access is forbidden in offline mode (missing local data?)")

    saved = {
        ("socket", "create_connection"): socket.create_connection,
        ("socket.socket", "connect"): socket.socket.connect,
        ("urllib.request", "urlopen"): urllib.request.urlopen,
    }
    socket.create_connection = blocked
    socket.socket.connect = blocked
    urllib.request.urlopen = blocked
    req_session = None
    try:
        import requests
        req_session = requests.Session.request
        requests.Session.request = blocked
    except Exception:  # requests not installed -> nothing to patch
        req_session = None
    try:
        yield counter
    finally:
        socket.create_connection = saved[("socket", "create_connection")]
        socket.socket.connect = saved[("socket.socket", "connect")]
        urllib.request.urlopen = saved[("urllib.request", "urlopen")]
        if req_session is not None:
            import requests
            requests.Session.request = req_session


def new_network_counter() -> _Counter:
    return _Counter()

"""Pytest session config for the CIGL CPU test suite.

Pin torch to a single thread: the tests run many tiny per-trial EEG ops / probe fits, and on the
shared many-core login node default multi-threading is pure dispatch overhead (and contends with
other users' load), making the suite 10-50x slower. Single-thread is both faster and deterministic
here. (The runners apply the same cap for their CPU subprocesses; the real GPU runs are unaffected.)
"""
import torch

torch.set_num_threads(1)

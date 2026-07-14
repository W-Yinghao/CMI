# C84FL2 Regression Verification

- `focused`: replacement job `896163`, 227 passed, 0 skipped, 0 deselected, stderr 0 bytes.
- `C65`: replacement job `896164`, 713 passed, 1 skipped, 3 deselected, stderr 0 bytes.
- `C23`: replacement job `896165`, 1124 passed, 1 skipped, 3 deselected, stderr 0 bytes.
- `full`: replacement job `896166`, 2048 passed, 1 skipped, 3 deselected, stderr 0 bytes.

Initial jobs `896157`-`896160` are preserved: each exposed six stale historical no-C84F-lock assertions. The replacement commit changed only those tests' lifecycle semantics. All jobs used `cpu-high`, 48 CPUs, 96 GiB and GPU 0. The sole cumulative skip is finalized C78F; the three cumulative deselections are historical C79 authorization-state checks. Every stderr file is empty.

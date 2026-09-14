# Future Work - uv Workspace Migration

Issues identified during Sourcery-AI review that require deeper changes.

## 1. True Meta-Package Architecture

**Current state:** Root `pyproject.toml` includes `packages = ["opentps_core/opentps", "opentps_gui/opentps"]` which bundles code from both subprojects into the root wheel.

**Issue:** This duplicates what opentps-core and opentps-gui provide, causing potential overlapping installs.

**Desired state:** Root project is a thin meta-package with only dependencies, no code. opentps-core and opentps-gui each ship their own `opentps` namespace packages.

**Blockers:**
- Removing the packages config breaks hatchling builds ("Unable to determine which files to ship")
- Requires proper namespace package setup in opentps_core and opentps_gui
- Entry point `opentps = "opentps.gui.main:run"` needs the namespace to be available

**Work required:**
- Configure opentps_core and opentps_gui as proper namespace packages
- Test that `pip install opentps` correctly pulls in both and entry point works
- Update documentation for recommended install pattern

## 2. Dockerfile and CI Alignment

**Current state:**
- Dockerfile: `uv pip install --system -e . -e opentps_core -e opentps_gui`
- CI: `pip install -e . -e opentps_core -e opentps_gui --no-deps`

**Issue:** Different install methods may not respect uv.lock consistently, affecting reproducibility.

**Work required:**
- Decide on canonical install method (uv sync vs uv pip install)
- Align Dockerfile and CI to use same approach
- Ensure uv.lock is respected in both contexts

## 3. Workspace-Aware Version Checking

**Current state:** `CI/check_version.py` only reads root `pyproject.toml`.

**Issue:** Dependencies declared in `opentps_core/pyproject.toml` and `opentps_gui/pyproject.toml` are not validated.

**Work required:**
- Extend check_version.py to discover workspace members from `[tool.uv.workspace]`
- Aggregate dependencies from all member pyproject.toml files
- Handle potential version conflicts between workspace members

---

*Documented during feat/uv_workspace MR review, January 2026*

#!/usr/bin/env python3
"""Unified version checker for PEP 621 and Poetry formats."""

import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from importlib.metadata import version as get_installed_version, PackageNotFoundError

TOML_PATH = Path("pyproject.toml")


def main():
    with open(TOML_PATH, "rb") as f:
        data = tomllib.load(f)

    # Detect format
    has_pep621 = "project" in data
    has_poetry = "poetry" in data.get("tool", {})

    if has_pep621:
        from packaging.specifiers import SpecifierSet
        from packaging.requirements import Requirement

        deps = {}
        for dep_str in data["project"].get("dependencies", []):
            req = Requirement(dep_str)
            if req.specifier:
                deps[req.name.lower()] = req.specifier

        python_req = data["project"].get("requires-python")
        if python_req:
            deps["python"] = SpecifierSet(python_req)

    elif has_poetry:
        from poetry.core.constraints.version import parse_constraint, Version

        deps = {}
        for pkg, constraint in data["tool"]["poetry"].get("dependencies", {}).items():
            if pkg.lower() == "python":
                continue
            if isinstance(constraint, dict):
                constraint = constraint.get("version", "*")
            if constraint and constraint != "*":
                deps[pkg.lower()] = parse_constraint(constraint)
    else:
        print("No [project] or [tool.poetry] section found")
        sys.exit(1)

    # Check versions
    mismatches = []
    for pkg, constraint in deps.items():
        if pkg == "python":
            installed = ".".join(map(str, sys.version_info[:3]))
        else:
            try:
                installed = get_installed_version(pkg)
            except PackageNotFoundError:
                mismatches.append(f"{pkg}: not installed")
                continue

        # Check satisfaction
        if has_pep621:
            if installed not in constraint:
                mismatches.append(f"{pkg}: {installed} not in {constraint}")
        else:
            from poetry.core.constraints.version import Version
            if not constraint.allows(Version.parse(installed)):
                mismatches.append(f"{pkg}: {installed} not in {constraint}")

    if mismatches:
        print("Version mismatches:")
        for m in mismatches:
            print(f"  {m}")
        sys.exit(1)

    print("All versions match.")
    sys.exit(0)


if __name__ == "__main__":
    main()

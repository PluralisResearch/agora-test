"""Current-tree code evidence, re-verified by every engine phase.

The lock pins the pithos source-tree digest at lock time; plan, worker,
finalize, verify, and publish all recompute the digest of the RUNNING tree
and refuse to operate on a mismatch — bytes produced or certified under one
code state are never attributed to another. The git revision pin remains
provenance (it may be unavailable from an installed wheel); the tree digest
is the enforced evidence.
"""

from __future__ import annotations

import hashlib
import os


_BUILD_MODULES = ("__init__.py", "corpus.py", "errors.py", "manifest.py")


def _build_code_files(root: str) -> list[tuple[str, str]]:
    files: list[tuple[str, str]] = []
    for name in _BUILD_MODULES:
        path = os.path.join(root, name)
        if os.path.isfile(path):
            files.append((name, path))
    build_root = os.path.join(root, "build")
    for dirpath, dirnames, filenames in os.walk(build_root):
        dirnames.sort()
        for filename in sorted(filenames):
            if filename.endswith(".py"):
                path = os.path.join(dirpath, filename)
                relative = os.path.relpath(path, root).replace(os.sep, "/")
                files.append((relative, path))
    return sorted(files)


def _build_tree_sha256(root: str) -> str:
    digest = hashlib.sha256()
    for relative, path in _build_code_files(root):
        encoded_path = relative.encode("utf-8")
        with open(path, "rb") as stream:
            content = stream.read()
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def pithos_tree_sha256() -> str:
    """Digest code that can affect corpus-build bytes or certification.

    The build package plus its shared corpus, manifest, error, and package
    modules are framed by path and content length before hashing. Runtime-only
    reader/cache changes do not invalidate an in-progress producer build.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../pithos
    return _build_tree_sha256(root)

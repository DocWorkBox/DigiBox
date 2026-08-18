# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Artifact resolution from local storage or HuggingFace Hub.

Each ArtifactManager is tied to a ``storage_root`` directory and a HuggingFace
repo.  Artifacts are served from ``storage_root`` when already present;
otherwise they are downloaded from HuggingFace and placed there, so the
directory acts as both a cache and the live artifact location.

This mirrors the original internal design: one root directory, predictable
layout, no hidden cache paths.

Storage layout (one manager per artifact group)::

    {storage_root}/{artifact_name}/{filename}     ← file artifacts
    {storage_root}/{artifact_name}/               ← directory artifacts

Example::

    mgr = ArtifactManager(
        repo_id="avaturn-live/avtr-1",
        revision="v1",
        artifacts={"appearance_extractor": ArtifactSource("runtime/appearance_extractor.onnx")},
        storage_root=Path("/my/storage/v1/renderer_runtime_artifacts"),
    )
    path = mgr.get_artifact_path("appearance_extractor")
    # → /my/storage/v1/renderer_runtime_artifacts/appearance_extractor/appearance_extractor.onnx
"""

from __future__ import annotations

import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, NamedTuple

LOG = logging.getLogger(__name__)


class ArtifactSource(NamedTuple):
    """A single artifact, optionally sourced from a different HuggingFace repo.

    ``path_in_repo``: local storage path relative to ``storage_root``.
    ``is_dir``: when True the entire directory subtree is downloaded.
    ``repo_id``: override source repo (default: the manager's ``repo_id``).
    ``repo_path``: path inside the override repo (default: same as ``path_in_repo``).
    ``revision``: override revision/branch (default: ``None`` → repo default branch).
    """

    path_in_repo: str
    is_dir: bool = False
    repo_id: str | None = None
    repo_path: str | None = None
    revision: str | None = None


class ArtifactManager:
    """Resolves artifacts from ``storage_root``, downloading from HF when absent.

    Args:
        repo_id:      HuggingFace repo, e.g. ``"avaturn-live/avtr-1"``.
        revision:     Git tag, branch, or commit hash.
        artifacts:    Logical name → ``ArtifactSource`` mapping.
        storage_root: Directory where artifacts are stored / downloaded to.
                      Set via ``AVTR1_LOCAL_STORAGE`` in the artifact manager.
    """

    def __init__(
        self,
        repo_id: str,
        revision: str,
        artifacts: Dict[str, ArtifactSource],
        *,
        storage_root: Path,
    ) -> None:
        self._repo_id = repo_id
        self._revision = revision
        self._artifacts = artifacts
        self._storage_root = Path(storage_root)

    def _artifact_path(self, name: str) -> Path:
        src = self._artifacts[name]
        return self._storage_root / src.path_in_repo

    def storage_path(self, name: str) -> Path:
        """Return where ``name`` is (or will be) stored, without downloading.

        Useful for build scripts that need to know the destination path before
        writing a file there.
        """
        if name not in self._artifacts:
            raise KeyError(f"Unknown artifact {name!r}. Available: {sorted(self._artifacts)}")
        return self._artifact_path(name)

    def get_artifact_path(self, name: str) -> Path:
        """Return the local path to ``name``, downloading from HF if absent."""
        if name not in self._artifacts:
            raise KeyError(f"Unknown artifact {name!r}. Available: {sorted(self._artifacts)}")
        path = self._artifact_path(name)
        if not path.exists():
            src = self._artifacts[name]
            repo = src.repo_id or self._repo_id
            rev = src.revision or self._revision
            LOG.info("Downloading %r from %s@%s", name, repo, rev)
            self._download(name)
        return path

    def _download(self, name: str) -> None:
        src = self._artifacts[name]
        if src.is_dir:
            self._download_dir(name, src)
        else:
            self._download_file(name, src)

    def _download_file(self, name: str, src: ArtifactSource) -> None:
        from huggingface_hub import hf_hub_download

        repo_id = src.repo_id or self._repo_id
        repo_path = src.repo_path or src.path_in_repo
        revision = src.revision or self._revision

        hf_path = Path(hf_hub_download(
            repo_id=repo_id,
            filename=repo_path,
            revision=revision,
            repo_type="model",
        ))
        dest = self._artifact_path(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(hf_path, dest)

    def _download_dir(self, name: str, src: ArtifactSource) -> None:
        from huggingface_hub import snapshot_download

        repo_id = src.repo_id or self._repo_id
        repo_path = src.repo_path or src.path_in_repo
        revision = src.revision or self._revision

        snapshot_root = Path(snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"{repo_path}/**", f"{repo_path}/*"],
            revision=revision,
            repo_type="model",
        ))
        src_dir = snapshot_root / repo_path
        dest_dir = self._artifact_path(name)
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(src_dir, dest_dir)

    def ensure_all_artifacts(self, workers: int = 4) -> None:
        """Pre-fetch all artifacts in parallel. No-op for already-present artifacts."""
        names = list(self._artifacts.keys())
        LOG.info("Ensuring %d artifacts in %s", len(names), self._storage_root)

        if workers <= 1 or len(names) <= 1:
            for name in names:
                self.get_artifact_path(name)
            return

        try:
            import tqdm as _tqdm
            _tqdm.tqdm.get_lock()
        except Exception:
            pass

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self.get_artifact_path, name): name for name in names}
            for fut in as_completed(futures):
                name = futures[fut]
                if exc := fut.exception():
                    raise RuntimeError(f"Failed to download artifact {name!r}") from exc

        LOG.info("All artifacts ready")


__all__ = ["ArtifactSource", "ArtifactManager"]

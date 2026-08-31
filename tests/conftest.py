"""Shared fixtures, including an in-memory stand-in for the Docker SDK.

The sandbox is exercised against a fake client rather than a real daemon so the suite
runs in CI. Real-daemon behaviour is verified separately; what these fakes protect is
the layer verity-core actually owns: how limits are passed, how commands are wrapped,
how output is decoded, and how snapshots and cleanup are sequenced.
"""

from __future__ import annotations

import io
import logging
import tarfile
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import pytest
from docker.errors import NotFound

from verity_core.logs import NAMESPACE

CommandHandler = Callable[[str, dict[str, bytes]], tuple[int, str, str]]


@pytest.fixture(autouse=True)
def isolate_logging() -> Iterator[None]:
    """Undo any change a test makes to the ``verity_core`` logger.

    The CLI configures logging as a side effect of running a command, which would
    otherwise leak handlers and levels into every test that ran after it.
    """
    logger = logging.getLogger(NAMESPACE)
    handlers = list(logger.handlers)
    level, propagate = logger.level, logger.propagate
    try:
        yield
    finally:
        logger.handlers = handlers
        logger.setLevel(level)
        logger.propagate = propagate


def _default_handler(command: str, files: dict[str, bytes]) -> tuple[int, str, str]:
    return 0, "", ""


class FakeImage:
    def __init__(self, image_id: str) -> None:
        self.id = image_id


class FakeContainer:
    """Records how it was created and answers execs from a test-supplied handler."""

    def __init__(self, container_id: str, image: str, kwargs: dict[str, Any]) -> None:
        self.id = container_id
        self.short_id = container_id[:12]
        self.image = image
        self.create_kwargs = kwargs
        self.files: dict[str, bytes] = {}
        self.commands: list[str] = []
        self.raw_commands: list[Sequence[str]] = []
        self.removed = False
        self.handler: CommandHandler = _default_handler

    def exec_run(
        self,
        cmd: Sequence[str],
        *,
        demux: bool = False,
        workdir: str | None = None,
        user: str = "",
        environment: dict[str, str] | None = None,
    ) -> tuple[int, tuple[bytes | None, bytes | None]]:
        self.raw_commands.append(list(cmd))
        # The runner wraps every command as
        # ["timeout", "--kill-after=Ns", "Ns", "/bin/sh", "-c", <command>].
        inner = cmd[-1]
        self.commands.append(inner)
        exit_code, stdout, stderr = self.handler(inner, self.files)
        out = stdout.encode() if stdout else None
        err = stderr.encode() if stderr else None
        return exit_code, (out, err) if demux else (out or b"") + (err or b"")

    def put_archive(self, destination: str, data: bytes) -> bool:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r") as tar:
            for member in tar.getmembers():
                extracted = tar.extractfile(member)
                payload = b"" if extracted is None else extracted.read()
                prefix = destination.rstrip("/")
                self.files[f"{prefix}/{member.name}"] = payload
        return True

    def get_archive(self, path: str) -> tuple[Iterator[bytes], dict[str, Any]]:
        if path not in self.files:
            raise NotFound(f"no such file: {path}")
        payload = self.files[path]
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as tar:
            info = tarfile.TarInfo(name=path.rsplit("/", 1)[-1])
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        return iter([archive.getvalue()]), {"size": len(payload)}

    def commit(self, *, repository: str, tag: str) -> FakeImage:
        return FakeImage(f"sha256:{repository}-{tag}")

    def remove(self, *, force: bool = False) -> None:
        self.removed = True


class FakeImages:
    def __init__(self) -> None:
        self.removed: list[str] = []

    def remove(self, image_id: str, *, force: bool = False) -> None:
        self.removed.append(image_id)


class FakeContainers:
    def __init__(self, client: FakeDockerClient) -> None:
        self._client = client

    def run(self, image: str, **kwargs: Any) -> FakeContainer:
        if self._client.run_error is not None:
            raise self._client.run_error
        self._client.counter += 1
        container = FakeContainer(f"container{self._client.counter:04d}", image, kwargs)
        container.handler = self._client.handler
        self._client.created.append(container)
        return container


class FakeDockerClient:
    """Minimal stand-in for :class:`docker.DockerClient`."""

    def __init__(self, handler: CommandHandler | None = None) -> None:
        self.handler = handler or _default_handler
        self.created: list[FakeContainer] = []
        self.images = FakeImages()
        self.containers = FakeContainers(self)
        self.counter = 0
        self.run_error: Exception | None = None

    @property
    def last(self) -> FakeContainer:
        return self.created[-1]

    def set_handler(self, handler: CommandHandler) -> None:
        """Point this client, and any container it already made, at ``handler``."""
        self.handler = handler
        for container in self.created:
            container.handler = handler


@pytest.fixture
def docker_client() -> FakeDockerClient:
    return FakeDockerClient()

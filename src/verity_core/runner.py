"""Docker container lifecycle management for sandboxed environment execution.

Everything an audit observes happens inside a :class:`SandboxRunner`. Two design
choices here are load-bearing rather than cosmetic:

* **Networking is off by default.** Several audit axes measure whether an
  environment leaks or reaches outside its container. If the default were "network
  on", a leak measurement would silently describe the harness instead of the
  environment.
* **Resource limits are always applied.** Comparing two environments' behaviour is
  only meaningful when both ran under identical CPU, memory, and wall-clock ceilings.
"""

from __future__ import annotations

import contextlib
import io
import json
import shlex
import tarfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

import docker
from docker.errors import DockerException, NotFound

DEFAULT_WORKDIR = "/workspace"
DEFAULT_KEEPALIVE_COMMAND = ("sleep", "infinity")
TIMEOUT_EXIT_CODE = 124
"""Exit status coreutils ``timeout`` reports when the deadline passes."""

SIGKILL_EXIT_CODE = 137
"""128 + SIGKILL. Ambiguous: both ``--kill-after`` escalation and an OOM kill land here."""

KILL_GRACE_SECONDS = 5
"""Grace period between the SIGTERM at the deadline and the follow-up SIGKILL."""

__all__ = [
    "DEFAULT_WORKDIR",
    "ExecResult",
    "ResourceLimits",
    "SandboxError",
    "SandboxRunner",
]


class SandboxError(RuntimeError):
    """Raised when the sandbox cannot be created, reached, or operated on."""


@dataclass(slots=True)
class ResourceLimits:
    """Ceilings applied to every container this runner creates."""

    cpu_count: float = 2.0
    memory_limit: str = "4g"
    timeout_seconds: int = 600
    network_disabled: bool = True

    def __post_init__(self) -> None:
        if self.cpu_count <= 0:
            raise ValueError(f"cpu_count must be positive, got {self.cpu_count}")
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {self.timeout_seconds}")

    @property
    def nano_cpus(self) -> int:
        """``cpu_count`` in the billionths-of-a-CPU unit the Docker API expects."""
        return int(self.cpu_count * 1_000_000_000)


@dataclass(slots=True)
class ExecResult:
    """Captured result of one command run inside the sandbox."""

    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    command: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "timed_out": self.timed_out,
        }


def _as_shell_command(command: str | Sequence[str]) -> str:
    return command if isinstance(command, str) else shlex.join(command)


def _decode(stream: bytes | None) -> str:
    return "" if not stream else stream.decode("utf-8", errors="replace")


def _is_timeout(exit_code: int, duration: float, limit: int) -> bool:
    """Decide whether an exit status represents the deadline being hit.

    124 is conclusive. 137 is only counted when the command actually ran to the
    deadline, so that a container OOM kill (which shares the status but usually
    happens early) is not misreported as a hang.
    """
    if exit_code == TIMEOUT_EXIT_CODE:
        return True
    return exit_code == SIGKILL_EXIT_CODE and duration >= limit


@dataclass(slots=True)
class SandboxRunner:
    """Creates, drives, snapshots, and tears down one sandbox container.

    ``client`` may be supplied to inject a Docker client (tests pass a fake); when
    omitted the real daemon connection is opened lazily on first use, so importing
    this module never requires a running Docker daemon.
    """

    image: str
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    workdir: str = DEFAULT_WORKDIR
    environment: dict[str, str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=lambda: {"verity.sandbox": "1"})
    client: Any | None = None
    _container: Any | None = field(default=None, init=False, repr=False)
    _snapshot_images: list[str] = field(default_factory=list, init=False, repr=False)

    def __enter__(self) -> SandboxRunner:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.cleanup()

    @property
    def container(self) -> Any:
        """The live container, raising if the sandbox has not been started."""
        if self._container is None:
            raise SandboxError("sandbox is not running; call start() first")
        return self._container

    @property
    def is_running(self) -> bool:
        return self._container is not None

    def _get_client(self) -> Any:
        if self.client is None:
            try:
                self.client = docker.from_env()
            except DockerException as exc:
                raise SandboxError(f"cannot reach the Docker daemon: {exc}") from exc
        return self.client

    def start(self) -> Any:
        """Create and start the container, replacing any container already running."""
        if self._container is not None:
            self.stop()
        client = self._get_client()
        try:
            self._container = client.containers.run(
                self.image,
                command=list(DEFAULT_KEEPALIVE_COMMAND),
                detach=True,
                network_disabled=self.limits.network_disabled,
                mem_limit=self.limits.memory_limit,
                nano_cpus=self.limits.nano_cpus,
                working_dir=self.workdir,
                environment=dict(self.environment),
                labels=dict(self.labels),
                # Keeps a misbehaving environment from writing over its own limits.
                privileged=False,
            )
        except DockerException as exc:
            raise SandboxError(
                f"failed to start container from image {self.image!r}: {exc}"
            ) from exc
        return self._container

    def exec(
        self,
        command: str | Sequence[str],
        *,
        timeout: int | None = None,
        workdir: str | None = None,
        user: str | None = None,
    ) -> ExecResult:
        """Run ``command`` in the container and capture stdout, stderr, and exit code.

        The timeout is enforced *inside* the container by wrapping the command in
        coreutils ``timeout``, because the Docker exec API offers no client-side
        deadline; a hung command would otherwise block the audit indefinitely.

        ``timeout`` sends SIGTERM at the deadline and escalates to SIGKILL after
        :data:`KILL_GRACE_SECONDS`. Signalling in that order matters: killing outright
        reports exit 137, which is indistinguishable from an OOM kill, whereas the
        graceful path reports an unambiguous 124.
        """
        shell_command = _as_shell_command(command)
        limit = self.limits.timeout_seconds if timeout is None else timeout
        wrapped = [
            "timeout",
            f"--kill-after={KILL_GRACE_SECONDS}s",
            f"{limit}s",
            "/bin/sh",
            "-c",
            shell_command,
        ]

        started = time.monotonic()
        try:
            exit_code, output = self.container.exec_run(
                wrapped,
                demux=True,
                workdir=workdir or self.workdir,
                user=user or "",
                environment=dict(self.environment),
            )
        except DockerException as exc:
            raise SandboxError(f"exec failed for {shell_command!r}: {exc}") from exc
        duration = time.monotonic() - started

        stdout_bytes, stderr_bytes = output if isinstance(output, tuple) else (output, None)
        return ExecResult(
            exit_code=exit_code,
            stdout=_decode(stdout_bytes),
            stderr=_decode(stderr_bytes),
            duration_seconds=duration,
            timed_out=_is_timeout(exit_code, duration, limit),
            command=shell_command,
        )

    def write_file(self, path: str, content: str | bytes, *, mode: int = 0o644) -> None:
        """Place a file inside the container without shelling out.

        Used to deliver submissions and solution scripts, which may contain quotes or
        newlines that would not survive being echoed through a shell.
        """
        data = content.encode("utf-8") if isinstance(content, str) else content
        parent, _, name = path.rpartition("/")
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as tar:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mode = mode
            tar.addfile(info, io.BytesIO(data))
        archive.seek(0)
        destination = parent or "/"
        try:
            self.container.put_archive(destination, archive.getvalue())
        except DockerException as exc:
            raise SandboxError(f"failed to write {path!r} into the container: {exc}") from exc

    def read_file(self, path: str) -> bytes:
        """Read a file out of the container, raising :class:`SandboxError` if absent."""
        try:
            stream, _ = self.container.get_archive(path)
        except NotFound as exc:
            raise SandboxError(f"no such file in container: {path!r}") from exc
        except DockerException as exc:
            raise SandboxError(f"failed to read {path!r} from the container: {exc}") from exc

        archive = io.BytesIO(b"".join(stream))
        archive.seek(0)
        with tarfile.open(fileobj=archive, mode="r") as tar:
            member = next((m for m in tar.getmembers() if m.isfile()), None)
            if member is None:
                raise SandboxError(f"archive for {path!r} contained no regular file")
            extracted = tar.extractfile(member)
            return b"" if extracted is None else extracted.read()

    def snapshot(self) -> bytes:
        """Commit the container to an image and return an opaque restore token.

        The token is JSON rather than a bare image ID so the format can gain fields
        (an overlay-filesystem diff path, for instance) without breaking callers who
        stored tokens from an earlier version.
        """
        try:
            image = self.container.commit(
                repository="verity-snapshot",
                tag=f"{self.container.short_id}-{int(time.time() * 1000)}",
            )
        except DockerException as exc:
            raise SandboxError(f"failed to snapshot container: {exc}") from exc

        image_id = getattr(image, "id", None) or str(image)
        self._snapshot_images.append(image_id)
        return json.dumps(
            {
                "version": 1,
                "kind": "docker-commit",
                "image": image_id,
                "source_image": self.image,
                "workdir": self.workdir,
            }
        ).encode("utf-8")

    def restore(self, snap: bytes) -> Any:
        """Replace the running container with one booted from ``snap``."""
        try:
            token = json.loads(snap.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SandboxError("snapshot token is not valid verity snapshot JSON") from exc
        if token.get("kind") != "docker-commit":
            raise SandboxError(f"unsupported snapshot kind: {token.get('kind')!r}")
        image = token.get("image")
        if not image:
            raise SandboxError("snapshot token is missing an image reference")

        self.stop()
        previous_image, self.image = self.image, image
        try:
            return self.start()
        except SandboxError:
            self.image = previous_image
            raise

    def stop(self) -> None:
        """Remove the current container, leaving snapshot images intact."""
        container, self._container = self._container, None
        if container is None:
            return
        try:
            container.remove(force=True)
        except NotFound:
            pass
        except DockerException as exc:
            raise SandboxError(f"failed to remove container: {exc}") from exc

    def cleanup(self) -> None:
        """Remove the container and every image this runner committed.

        Best-effort by design: cleanup usually runs while an exception is already
        propagating, and masking that exception with a teardown failure would hide
        the finding the audit was about to report.
        """
        with contextlib.suppress(SandboxError):
            self.stop()

        client = self.client
        if client is None:
            self._snapshot_images.clear()
            return
        for image_id in self._snapshot_images:
            with contextlib.suppress(DockerException, NotFound):
                client.images.remove(image_id, force=True)
        self._snapshot_images.clear()

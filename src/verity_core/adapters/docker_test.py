"""Adapter for generic "container plus test command" tasks (R2E-Gym, SWE-Gym style).

The submission for these tasks is normally a patch against a checked-out repository,
so nothing runs it directly: ``apply_command`` is left to the manifest, since how a
patch is applied (``git apply``, ``patch -p1``, a repo-specific script) varies by
upstream. When it is set, a submission that fails to apply is recorded as a failure
rather than being handed to the tests.
"""

from __future__ import annotations

from verity_core.adapters.base import ContainerEnv

__all__ = ["DockerTestAdapter"]


class DockerTestAdapter(ContainerEnv):
    """Wraps an image-and-eval-command task as a :class:`~verity_core.env.VerityEnv`.

    Recognised manifest fields, beyond the shared ones:

    ``image``
        Prebuilt task image (required; Dockerfile builds are not wired up yet).
    ``test_command``
        Grading command (required). No convention exists to default to here.
    ``submission_path``
        Where the patch is written. Defaults to ``/workspace/submission.patch``.
    ``apply_command``
        Optional command that applies the patch before the tests run.
    ``gold_patch``
        Reference patch, accepted as an alias for ``gold_solution``.
    ``reset_before_verify``
        Defaults to true for this format; set false to grade a reused container.
    """

    default_domain = "code"
    default_submission_path = "/workspace/submission.patch"
    gold_aliases = ("gold_patch",)
    # These tasks are graded on the submission alone, so each verification gets a
    # clean container rather than one still carrying the previous patch.
    reset_before_verify = True

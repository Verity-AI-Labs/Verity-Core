"""Adapter for Terminal Wrench / Terminal-Bench tasks.

These tasks ship as a Docker image plus two conventions: the agent's work is a shell
script, and grading is a test script baked into the image. The submission is therefore
*executed* before the tests run, which is why this format has a default
``apply_command`` where the generic container format does not.
"""

from __future__ import annotations

from verity_core.adapters.base import ContainerEnv

__all__ = ["TerminalAdapter"]


class TerminalAdapter(ContainerEnv):
    """Wraps a Terminal-Bench style task as a :class:`~verity_core.env.VerityEnv`.

    Recognised manifest fields, beyond the shared ones:

    ``image``
        Prebuilt task image (required; Dockerfile builds are not wired up yet).
    ``test_command``
        Grading command. Defaults to Terminal-Bench's ``bash /tests/run-tests.sh``.
    ``submission_path``
        Where the candidate script is written. Defaults to ``/workspace/solution.sh``.
    ``apply_command``
        How the script runs before grading. Defaults to ``bash <submission_path>``.
    ``solution``
        Reference solution script, accepted as an alias for ``gold_solution``.
    ``reset_before_verify``
        Defaults to false, since the agent's container is the thing being graded. Set
        it true for independent repeated trials of the same task.
    """

    default_domain = "tool_use"
    default_submission_path = "/workspace/solution.sh"
    default_test_command = "bash /tests/run-tests.sh"
    # Terminal-Bench calls the reference script `solution.sh`, so corpus entries tend
    # to use a `solution` key; accept it rather than making them rename it.
    gold_aliases = ("solution",)

    def _default_apply_command(self) -> str:
        return f"bash {self.submission_path}"

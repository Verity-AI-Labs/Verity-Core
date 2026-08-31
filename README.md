# verity-core

The shared foundation for Verity Labs' RL environment auditing tools.

Verity Labs audits the quality of reinforcement learning training environments. Those
environments arrive from many upstream projects in incompatible formats, and all four
audit tools need to do the same handful of things with them: load them, run things
inside them, call models, collect verifier verdicts, and record results.

verity-core implements that once. The tools — **Verity-RedTeam**, **Verity-Signal**,
**Verity-Clean**, and **Verity-Stable** — and the **Verity-Corpus** manifest repo all
depend on this library, so none of them reimplements environment loading, sandboxing,
model access, or the scorecard format.

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). Docker is needed only to
run container-backed environments, not to import the library or run the test suite.

```bash
uv sync
```

## Run the tests

```bash
uv run pytest
```

Lint and formatting, the same checks CI runs:

```bash
uv run ruff check .
uv run ruff format --check .
```

## The components

### 1. `VerityEnv` protocol — `verity_core/env.py`

The universal interface every environment conforms to: `spec()`, `reset()`,
`step()`, `verify()`, `gold_solution()`, `snapshot()`, and `restore()`.

It is a `typing.Protocol`, not an abstract base class, so adapters conform
structurally and no upstream environment has to inherit from our class hierarchy.
Alongside it are the types that cross the interface: `TaskSpec`, `Observation`,
`StepResult`, and `RewardResult`.

`TaskSpec` pins a task to an exact upstream revision through its `source` and `commit`
fields. `RewardResult` keeps `reward` and `verdict` separate, because a partial-credit
verifier can return 0.6 while still failing the task, and several audit axes depend on
seeing that disagreement.

### 2. Adapters — `verity_core/adapters/`

One adapter per upstream format, each turning a manifest entry into a `VerityEnv`:

| Adapter | Upstream format |
| --- | --- |
| `VerifiersAdapter` | PrimeIntellect / `verifiers` rollout+reward environments |
| `TerminalAdapter` | Terminal Wrench / Terminal-Bench tasks |
| `DockerTestAdapter` | Generic container + test command (R2E-Gym, SWE-Gym style) |

`load_env()` reads the `format` field and returns the right one, so a tool never needs
to know which project a task came from:

```python
from verity_core import load_env

env = load_env(manifest_entry)  # -> VerityEnv
```

### 3. Sandbox runner — `verity_core/runner.py`

`SandboxRunner` owns the container lifecycle: create from an image under CPU, memory,
and wall-clock limits; execute commands and capture stdout, stderr, and exit codes;
move files in and out; snapshot via `docker commit` and restore; then clean up.

**Networking is disabled by default.** Several audit axes measure whether an
environment reaches outside its container, so a permissive default would mean those
measurements described the harness instead of the environment.

Timeouts are enforced inside the container with coreutils `timeout`, which sends
SIGTERM at the deadline and escalates to SIGKILL after a grace period. The ordering
matters: killing outright reports exit 137, which is indistinguishable from an OOM
kill, whereas the graceful path reports an unambiguous 124.

### 4. Model client — `verity_core/models.py`

`ModelClient` calls any OpenAI-compatible `/v1/chat/completions` endpoint, defaulting
to a local vLLM server at `http://localhost:8000/v1`. It adds two things audits need:

- **A disk response cache.** Re-running an audit must not re-sample the model, or two
  scorecards of the same environment stop being comparable. The cache key covers the
  endpoint as well as the model, messages, and sampling parameters, since two servers
  can serve different weights under the same model name.
- **A usage accumulator.** `client.total_usage` reports the tokens the session actually
  spent, so a scorecard can state what it cost. Cache hits are excluded.

The client is deliberately a thin `httpx` wrapper rather than a multi-provider
abstraction; `litellm` would add a dependency without simplifying a single
OpenAI-compatible endpoint.

### 5. Scorecard — `verity_core/scorecard.py`

`Scorecard` is the audit output format: one `AxisValue` per rubric axis, each recording
the value, which tool produced it, the raw supporting evidence, and notes. It
serializes to JSON and renders to markdown.

The 13 scored axes are **V1–V7** and **U1, U2, U3, U4, U6, U7**. **U5 (transfer value)
is excluded** because it is the downstream outcome the other axes are meant to predict;
scoring it here would leak the label into the measurements used to predict it.

Every axis exists on a scorecard from construction, unscored ones included, so a reader
can always distinguish "we measured zero" from "we did not look".

### 6. Configuration — `verity_core/config.py`

`load_config()` resolves `VerityConfig` per field, with `verity.yaml` values winning
over `VERITY_*` environment variables, which win over the defaults. Resolving per field
means a config file that only pins `model_name` still picks up a `VERITY_CACHE_DIR` set
by CI.

```yaml
# verity.yaml
model_base_url: http://localhost:8000/v1
model_name: Qwen/Qwen2.5-7B-Instruct
cache_dir: .verity_cache
results_dir: results
docker_timeout: 600
docker_memory_limit: 4g
docker_network_disabled: true
```

Unknown keys are rejected rather than ignored, so a typo like `docker_memroy_limit`
cannot leave an audit running with limits the operator believes they overrode.

## Usage

```python
from verity_core import ModelClient, Scorecard, load_config, load_env

config = load_config()
config.ensure_dirs()

client = ModelClient.from_config(config)

manifest_entry = {
    "id": "terminal-bench/hello-world",
    "format": "terminal-bench",
    "image": "verity/hello-world:latest",
    "source": "https://github.com/laude-institute/terminal-bench",
    "commit": "abc1234",
    "instructions": "Write a script that creates /workspace/done.",
    "test_command": "bash /tests/run-tests.sh",
    "solution": "touch /workspace/done\n",
    "limits": {"cpu_count": 2, "memory_limit": "4g", "timeout_seconds": 60},
}

env = load_env(manifest_entry)
spec = env.spec()

try:
    response = client.complete(
        config.model_name,
        [{"role": "user", "content": spec.instructions}],
    )
    result = env.verify(response.content)

    # A second trial on this format needs a clean container, or the file the first
    # submission created would score the second one too.
    gold = env.gold_solution()
    if gold:
        env.reset()
        gold_result = env.verify(gold)
    else:
        gold_result = None
finally:
    env.close()

scorecard = Scorecard(env_id=spec.id)
scorecard.set_axis(
    "V1",
    value=result.reward,
    tool="verity-core-example",
    evidence={
        "verdict": result.verdict,
        "verifier_logs": result.verifier_logs,
        "gold_passes": None if gold_result is None else gold_result.verdict,
        "tokens": client.total_usage.to_dict(),
    },
    notes="single-sample smoke check",
)

scorecard.to_json(config.results_dir / f"{spec.id.replace('/', '_')}.json")
print(scorecard.to_markdown())
```

## Manifest fields

Every manifest entry needs `id` and `format`. Corpus entries are also expected to carry
`source` and `commit`, which are what make an audit reproducible.

**Shared:** `domain` (`browser`, `gui`, `tool_use`, `code`, `math`, `other`),
`reward_type` (`binary` or `partial`), `instructions`, `has_gold`, `gold_solution`.

**Container formats** (`terminal`, `docker_test`): `image`, `test_command`,
`submission_path`, `apply_command`, `setup_commands`, `gold_solution_path`,
`reset_before_verify`, and `limits` (`cpu_count`, `memory_limit`, `timeout_seconds`,
`network_disabled`).

`reset_before_verify` is the one field worth reading closely. It defaults to false for
`terminal`, whose tasks are graded on the container the agent worked in, and true for
`docker_test`, whose tasks are graded on the submission alone and would otherwise carry
a previous trial's patch into the next one.

**`verifiers` format:** `reward` and `rollout` as `"package.module:function"`
references, `task` for the config passed to them, and `pass_threshold` for the reward
at which the verdict becomes true.

Format names are matched loosely: `terminal-bench`, `terminal_wrench`, and `tbench` all
select `TerminalAdapter`; `prime` and `swe-gym` map to their adapters too. Use
`register_adapter()` to add a format without changing verity-core.

## Not yet implemented

The interfaces and types are complete, and the container lifecycle works end to end
against a real daemon. Two internals are still stubbed and marked with `TODO` in the
source:

- **Building images from a `dockerfile` field.** Manifests must currently supply a
  prebuilt `image`; a Dockerfile-only entry raises a clear `ManifestError`.
- **Partial-credit grading.** Verifier output is not yet parsed per test, so a
  `partial` reward type falls back to a binary pass/fail. When it does, the scorecard
  evidence records `grading: binary_fallback` rather than passing it off silently.

## License

Apache-2.0. See [LICENSE](LICENSE).

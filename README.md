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

It also carries the scaffolding an audit needs at corpus scale: loading a directory of
manifests, running an audit function across all of them with failures isolated and the
run resumable, aggregating the resulting scorecards into a corpus-level defect report,
and logging enough to reconstruct what happened at environment 147.

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
`step()`, `verify()`, `gold_solution()`, `snapshot()`, and `restore()`, plus `close()`
and the context-manager pair `__enter__` / `__exit__`.

Teardown is part of the protocol, not an adapter detail, so a tool holding a generic
`VerityEnv` can always release what is behind it:

```python
with load_env(entry) as env:
    result = env.verify(submission)
# the container, if there was one, is gone here
```

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

The precedence rule lives in one place, the `LAYERS` tuple, which the resolver folds in
order rather than relying on which dict gets updated into which:

```python
LAYERS: tuple[str, ...] = ("defaults", "environment", "file")
```

`resolve_config()` additionally reports which layer supplied each value, which is what
`verity-core config` prints when you need to know why a setting is what it is.

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

### 7. Corpus loader — `verity_core/corpus.py`

`load_corpus()` turns a directory of YAML manifests into a list of validated manifest
entry dicts, each ready for `load_env()`. It touches neither Docker nor any environment:
counting how many browser tasks a corpus holds should not start a container.

```python
from verity_core import corpus_stats, load_corpus

browser = load_corpus("../Verity-Corpus/manifests/", domain="browser")
for entry in browser:
    with load_env(entry) as env:
        ...

stats = corpus_stats(load_corpus("../Verity-Corpus/manifests/"))
print(stats.total, stats.by_domain, stats.gold_coverage)
```

Non-YAML files are skipped, empty manifests are ignored with a warning, and `id` /
`format` are required. Format filtering is alias-aware, so `format="terminal"` also
matches entries written as `terminal-bench`. Entries come back sorted by `id` so that
repeated runs visit environments in the same order, which batch resumption depends on.

Duplicate ids are always an error, even in `strict=False` mode: scorecard filenames
derive from the id, so a duplicate would have one environment's results overwrite
another's.

### 8. Batch runner — `verity_core/batch.py`

`run_batch()` runs a tool's audit function across a corpus and collects the outcomes. A
single environment crashing is recorded and stepped over rather than allowed to end the
run, since broken environments are what an audit exists to find.

```python
from verity_core import load_corpus, run_batch


def audit(entry):
    with load_env(entry) as env:
        result = env.verify(env.gold_solution() or "")
    card = Scorecard(env_id=entry["id"], metadata={"domain": entry.get("domain")})
    card.set_axis("V1", 0.0 if result.verdict else 1.0, "my-tool")
    return card


batch = run_batch(
    load_corpus("../Verity-Corpus/manifests/"),
    audit,
    results_dir="results/",
    resume=True,
    model_client=client,
)
print(batch.succeeded, batch.failed, batch.skipped, batch.total_tokens)
for failure in batch.failures:
    print(failure.env_id, failure.error_type, failure.error)
```

With `resume=True` the runner skips environments that already have a scorecard in the
results directory, so a run that dies at environment 180 of 200 picks up where it left
off instead of starting over. Ids are read from inside each scorecard rather than
inferred from filenames, and unreadable files are ignored so that a scorecard truncated
by the crash you are resuming from gets re-audited.

Returning `None` from the audit function records a skip rather than a failure, which is
how a tool declares an environment not applicable. `Ctrl-C` stops the run and returns
what completed, with `interrupted` set.

### 9. Reporting — `verity_core/reporting.py`

`build_corpus_report()` aggregates a directory of scorecards into the corpus-level
result: per-axis statistics, defect rates, domain breakdown, and the most-flagged
environments, rendered as JSON and Markdown.

```python
from verity_core import build_corpus_report

report = build_corpus_report("results/", threshold=0.5, output_dir="reports/")
print(report.defect_rate)  # fraction flagged on at least one validity axis
print(report.axis_stats["V1"].mean)
print(report.to_markdown())
```

An axis nobody measured is counted under `missing`, never averaged in as zero, so a thin
run reads as low coverage rather than a clean corpus.

Flagging direction is a parameter, not an assumption. The default flags a score at or
above the threshold, matching a rubric where a higher score is a more serious defect;
pass `direction="below"` for a rubric scored the other way.

### 10. CLI — `verity_core/cli.py`

Installed as `verity-core`. A convenience layer for the work around an audit; tools
still import the Python API directly.

```bash
verity-core corpus ../Verity-Corpus/manifests/     # counts by domain, format, gold coverage
verity-core corpus manifests/ --domain browser --json
verity-core audit manifests/hello-world.yaml --results-dir results/
verity-core report results/ --threshold 0.5 --output reports/
verity-core config                                 # resolved values, annotated by source
```

`verity-core config` answers the question you actually have when a run misbehaves:

```
FIELD                    VALUE                     FROM
model_base_url           http://localhost:8000/v1  defaults
model_name               from-file                 file
docker_timeout           42                        file
docker_memory_limit      16g                       environment
```

`verity-core audit` fills in only what verity-core can measure alone: whether the
environment's own verifier accepts its own reference solution. That lands on **V1**,
scored as a defect indicator — `1.0` means the verifier *rejected* its own gold
solution, which matches the direction the report flags on. The other twelve axes are
left unscored for the four tools to fill in.

Summaries go to stdout and logs to stderr, so `--json` output stays safe to pipe at any
log level. Add `--log-level info` to watch an audit progress.

### 11. Logging — `verity_core/logs.py`

Every module logs through the `verity_core` namespace using the standard library only,
so a tool can set verbosity for the whole library from its own entry point:

```python
import logging
from verity_core import configure_logging

configure_logging(logging.INFO)  # our defaults, to stderr
logging.getLogger("verity_core").setLevel(logging.DEBUG)  # or configure it yourself
```

Messages are written as `event key=value` so a 200-environment run stays greppable:

```
2026-08-31T11:03:50 INFO  verity_core.runner | container started id=2aae1f96675b image=python:3.13-slim network_disabled=True
2026-08-31T11:03:50 INFO  verity_core.adapters.base | verify complete env_id=docker/hello-world verdict=True reward=1.000 exit_code=0 duration=0.03s
```

`DEBUG` covers routine operations (every exec, every cache lookup), `INFO` lifecycle
events (environment loaded, container started, scorecard written), `WARNING` recoverable
problems that change how a result should be read (command timed out, submission would
not apply, partial-credit grader fell back to binary, cache entry unreadable), and
`ERROR` failures (daemon unreachable, container would not start, manifest invalid).

The library installs only a `NullHandler`, so importing it never configures logging for
its host.

## Usage

```python
from verity_core import (
    ModelClient,
    Scorecard,
    build_corpus_report,
    load_config,
    load_env,
)
from verity_core.scorecard import scorecard_path

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

with load_env(manifest_entry) as env:
    spec = env.spec()

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

scorecard = Scorecard(env_id=spec.id, metadata={"domain": spec.domain})
scorecard.set_axis(
    "V1",
    # Scored as a defect: 1.0 means the environment rejected its own reference
    # solution. This is the direction the reporting layer flags on.
    value=None if gold_result is None else (0.0 if gold_result.verdict else 1.0),
    tool="verity-core-example",
    evidence={
        "model_verdict": result.verdict,
        "model_reward": result.reward,
        "gold_verdict": None if gold_result is None else gold_result.verdict,
        "verifier_logs": result.verifier_logs,
        "tokens": client.total_usage.to_dict(),
    },
    notes="single-sample smoke check",
)

scorecard.to_json(scorecard_path(config.results_dir, spec.id))
print(scorecard.to_markdown())
```

Aggregating a directory of those scorecards into the corpus-level result:

```python
report = build_corpus_report(config.results_dir, output_dir="reports/")
print(f"defect rate: {report.defect_rate:.1%} of {report.total_scorecards} environments")
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
`register_adapter()` to add a format without changing verity-core. Registration mutates
module-level dicts and is not thread-safe; call it once during single-threaded startup.

## Not yet implemented

The interfaces and types are complete, and the container lifecycle works end to end
against a real daemon. Two internals are still stubbed and marked with `TODO` in the
source:

- **Building images from a `dockerfile` field.** Manifests must currently supply a
  prebuilt `image`; a Dockerfile-only entry raises a clear `ManifestError`.
- **Partial-credit grading.** Verifier output is not yet parsed per test, so a
  `partial` reward type falls back to a binary pass/fail. When it does, the scorecard
  evidence records `grading: binary_fallback` rather than passing it off silently, and a
  `WARNING` is logged.

Two deliberate scope limits, not stubs:

- **The batch runner is sequential.** Environments are audited one at a time. The
  parallel worker pool comes later, and `register_adapter()` and the response cache are
  annotated with what that will require.
- **`verity-core audit` fills in one axis.** The other twelve belong to the four tools.

## License

Apache-2.0. See [LICENSE](LICENSE).

"""Kaggle entry cell: run all experiments and push recoverable results to GitHub.

Before running, add a private Kaggle secret named ``GITHUB_TOKEN`` with
Contents: Read and write permission for the target repository. The token is
passed through an HTTP authorization header and is never stored in the remote
URL or printed by this script.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from kaggle_secrets import UserSecretsClient
import base64
import hashlib
import json
import os
import subprocess
import sys


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

REPOSITORY = "1101909/FIR-MIC-MMREC-Cold"
SOURCE_BRANCH = "agent/fix-semco-evaluator"
# Keep grid-search results separate from legacy/non-tuned result artifacts.
RESULTS_BRANCH = "kaggle-results-gridsearch-v1"

DATA_DIR = "/kaggle/input/datasets/toanktx/mmrec-cold"
REPO_DIR = Path("/kaggle/working/FIR-MIC-MMREC-Cold")

DATASETS = ["baby", "clothing", "sports"]
SEEDS = list(range(2022, 2032))
MAX_EPOCHS = 20
BATCH_SIZE = 256

GRID_INTENTS = [32, 64]
GRID_DIMS = [64]
GRID_LEARNING_RATES = [0.0005, 0.001]
GRID_WEIGHT_DECAYS = [0.0001]
GRID_DROPOUTS = [0.15]
GRID_ROUTER_DROPOUTS = [0.10]
GRID_PATIENCE = 5

RUN_CONTROLLED_CONTENT_BASELINES = True
RUN_OFFICIAL_COLD_BASELINES = True
RUN_EXTENDED_ABLATIONS = True


def run(command, cwd=None, env=None, check=True):
    command = [str(value) for value in command]
    print("\n+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=cwd, env=env, check=check)


def output(command, cwd=None, env=None):
    return subprocess.check_output(
        [str(value) for value in command], cwd=cwd, env=env, text=True
    ).strip()


# ---------------------------------------------------------------------------
# Clone/reuse and authenticate
# ---------------------------------------------------------------------------

if not REPO_DIR.exists():
    run([
        "git", "clone", "--branch", SOURCE_BRANCH, "--single-branch",
        f"https://github.com/{REPOSITORY}.git", REPO_DIR,
    ])
elif not (REPO_DIR / ".git").exists():
    raise RuntimeError(f"{REPO_DIR} exists but is not a Git repository")
else:
    print(f"Reusing repository: {REPO_DIR}")

token = UserSecretsClient().get_secret("GITHUB_TOKEN")
if not token:
    raise RuntimeError("Kaggle Secret GITHUB_TOKEN was not found")

encoded_auth = base64.b64encode(
    f"x-access-token:{token}".encode("utf-8")
).decode("utf-8")
git_env = os.environ.copy()
git_env["GIT_CONFIG_COUNT"] = "1"
git_env["GIT_CONFIG_KEY_0"] = "http.https://github.com/.extraheader"
git_env["GIT_CONFIG_VALUE_0"] = f"AUTHORIZATION: basic {encoded_auth}"

run(["git", "config", "user.name", "Kaggle GPU Runner"], cwd=REPO_DIR)
run([
    "git", "config", "user.email", "kaggle-runner@users.noreply.github.com",
], cwd=REPO_DIR)


# ---------------------------------------------------------------------------
# Synchronize the result branch with the current source code
# ---------------------------------------------------------------------------

run([
    "git", "fetch", "origin",
    f"{SOURCE_BRANCH}:refs/remotes/origin/{SOURCE_BRANCH}",
], cwd=REPO_DIR, env=git_env)

remote_results = subprocess.run([
    "git", "ls-remote", "--exit-code", "--heads", "origin",
    f"refs/heads/{RESULTS_BRANCH}",
], cwd=REPO_DIR, env=git_env)

if remote_results.returncode == 0:
    run([
        "git", "fetch", "origin",
        f"{RESULTS_BRANCH}:refs/remotes/origin/{RESULTS_BRANCH}",
    ], cwd=REPO_DIR, env=git_env)
    local_results = subprocess.run([
        "git", "show-ref", "--verify", "--quiet",
        f"refs/heads/{RESULTS_BRANCH}",
    ], cwd=REPO_DIR)
    if local_results.returncode == 0:
        run(["git", "switch", RESULTS_BRANCH], cwd=REPO_DIR)
        run([
            "git", "merge", "--ff-only", f"origin/{RESULTS_BRANCH}",
        ], cwd=REPO_DIR)
    else:
        run([
            "git", "switch", "-c", RESULTS_BRANCH, "--track",
            f"origin/{RESULTS_BRANCH}",
        ], cwd=REPO_DIR)
    # Results remain on their own branch while always receiving the latest
    # experiment runner fixes from the source branch.
    run([
        "git", "merge", "--no-edit", f"origin/{SOURCE_BRANCH}",
    ], cwd=REPO_DIR)
else:
    run([
        "git", "switch", "-c", RESULTS_BRANCH, f"origin/{SOURCE_BRANCH}",
    ], cwd=REPO_DIR)

run(["git", "push", "-u", "origin", RESULTS_BRANCH], cwd=REPO_DIR, env=git_env)
source_commit = output([
    "git", "rev-parse", f"origin/{SOURCE_BRANCH}",
], cwd=REPO_DIR)


# ---------------------------------------------------------------------------
# Dependencies and reproducible run identity
# ---------------------------------------------------------------------------

run([
    sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt",
], cwd=REPO_DIR)

run_configuration = {
    "source_branch": SOURCE_BRANCH,
    "source_commit": source_commit,
    "max_epochs": MAX_EPOCHS,
    "batch_size": BATCH_SIZE,
    "grid_intents": GRID_INTENTS,
    "grid_dims": GRID_DIMS,
    "grid_learning_rates": GRID_LEARNING_RATES,
    "grid_weight_decays": GRID_WEIGHT_DECAYS,
    "grid_dropouts": GRID_DROPOUTS,
    "grid_router_dropouts": GRID_ROUTER_DROPOUTS,
    "grid_patience": GRID_PATIENCE,
    "controlled_content_baselines": RUN_CONTROLLED_CONTENT_BASELINES,
    "official_cold_baselines": RUN_OFFICIAL_COLD_BASELINES,
    "extended_ablations": RUN_EXTENDED_ABLATIONS,
}
run_signature = hashlib.sha256(
    json.dumps(run_configuration, sort_keys=True).encode("utf-8")
).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Commit and push all small, recoverable artifacts after every seed
# ---------------------------------------------------------------------------

def upload_results(dataset, seed, exit_code):
    results_dir = REPO_DIR / "results"
    if not results_dir.exists():
        print("No results directory found")
        return

    allowed_suffixes = {".json", ".csv", ".md", ".txt"}
    result_files = [
        str(path.relative_to(REPO_DIR))
        for path in results_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed_suffixes
    ]
    if not result_files:
        print("No result artifacts to upload")
        return

    run(["git", "add", "-f", "--", *result_files], cwd=REPO_DIR)
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO_DIR)
    if staged.returncode == 0:
        print("No new result changes to commit")
        return

    status = "complete" if exit_code == 0 else f"partial-exit-{exit_code}"
    run([
        "git", "commit", "-m",
        f"Add {dataset} seed {seed} grid-search results ({status})",
    ], cwd=REPO_DIR)
    run(["git", "push", "origin", RESULTS_BRANCH], cwd=REPO_DIR, env=git_env)


checkpoint_dir = REPO_DIR / "results" / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Run baselines, tuned FIR-MIC, RQ3 and evidence aggregation
# ---------------------------------------------------------------------------

for dataset in DATASETS:
    for seed in SEEDS:
        marker = checkpoint_dir / dataset / f"seed_{seed}.json"
        if marker.exists():
            marker_data = json.loads(marker.read_text(encoding="utf-8"))
            if (marker_data.get("status") == "completed"
                    and marker_data.get("run_signature") == run_signature):
                print(f"\nSKIP COMPLETED: dataset={dataset}, seed={seed}", flush=True)
                continue
            raise RuntimeError(
                f"Existing completion marker {marker} belongs to a different "
                "experiment configuration. Use a new RESULTS_BRANCH instead "
                "of mixing incompatible results."
            )

        command = [
            sys.executable, "kaggle/run_gpu.py",
            "--data-dir", DATA_DIR,
            "--datasets", dataset,
            "--seeds", seed,
            "--epochs", MAX_EPOCHS,
            "--batch-size", BATCH_SIZE,
            "--grid-intents", *GRID_INTENTS,
            "--grid-dims", *GRID_DIMS,
            "--grid-learning-rates", *GRID_LEARNING_RATES,
            "--grid-weight-decays", *GRID_WEIGHT_DECAYS,
            "--grid-dropouts", *GRID_DROPOUTS,
            "--grid-router-dropouts", *GRID_ROUTER_DROPOUTS,
            "--grid-patience", GRID_PATIENCE,
            "--resume",
        ]
        if RUN_CONTROLLED_CONTENT_BASELINES:
            command.append("--run-controlled-content-baselines")
        if RUN_OFFICIAL_COLD_BASELINES:
            command.append("--run-official-cold-baselines")
        if RUN_EXTENDED_ABLATIONS:
            command.append("--extended-ablations")

        print("\n" + "=" * 80)
        print(f"RUNNING dataset={dataset}, seed={seed}, signature={run_signature}")
        print("=" * 80, flush=True)
        completed = run(command, cwd=REPO_DIR, check=False)

        if completed.returncode == 0:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({
                "dataset": dataset,
                "seed": seed,
                "status": "completed",
                "run_signature": run_signature,
                "run_configuration": run_configuration,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2), encoding="utf-8")

        upload_results(dataset, seed, completed.returncode)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Run failed for dataset={dataset}, seed={seed}, "
                f"exit code={completed.returncode}. Partial JSON/CSV and grid "
                "trial checkpoints were pushed. Run this cell again to resume."
            )


print("\n" + "=" * 80)
print("ALL EXPERIMENTS COMPLETED")
print("=" * 80)
print(
    f"Results: https://github.com/{REPOSITORY}/tree/{RESULTS_BRANCH}/results"
)

"""Modal planning script for robust-world-model-planning.

Usage:
    modal run modal_plan.py::download_all                                           # all checkpoints + datasets (parallel)
    modal run modal_plan.py::download_pretrained                                    # pretrained (DINO-WM) checkpoints, all envs
    modal run modal_plan.py::download_checkpoints --env-name pusht --method online  # one env/method from GDrive
    modal run modal_plan.py::download_datasets --env-name pusht                     # one dataset
    modal run modal_plan.py --env-name pusht --method pretrained                    # run planning (W&B always on)
    modal run modal_plan.py --env-name wall --method adversarial                    # run planning + W&B
"""

import modal

app = modal.App("robust-worldmodel-plan")

volume = modal.Volume.from_name("stableworldmodel")
ROBUST_WM_HOME = "/root/.robust_worldmodel"

# Pretrained DINO-WM checkpoints — all envs bundled in one archive on OSF
OSF_OUTPUTS_URL = "https://osf.io/download/xvzs4/?view_only=a56a296ce3b24cceaf408383a175ce28"

# Datasets from OSF
DATASET_URLS = {
    "pusht":      "https://osf.io/download/k2d8w/?view_only=a56a296ce3b24cceaf408383a175ce28",
    "point_maze": "https://osf.io/download/vr5gy/?view_only=a56a296ce3b24cceaf408383a175ce28",
    "wall":       "https://osf.io/download/49rnx/?view_only=a56a296ce3b24cceaf408383a175ce28",
}

ENV_CONFIGS = {
    "pusht": dict(
        plan_config="plan_pusht",
        gdrive_online="14rn3dD4ezLC4Qh5Xfwrlwe7_Wlxohecu",
        gdrive_adversarial="1le0G8zJYRJTz-2lyOFtn6QhCqE6cuZwo",
        dataset_dir="pusht_noise",
    ),
    "point_maze": dict(
        plan_config="plan_point_maze",
        gdrive_online="1bpX5TaW4DhgfKDj7iwQ5r2fk4teVCUiW",
        gdrive_adversarial="1jB24wVsw7dRy9PwpV0DhzcWQNmXMLtsl",
        dataset_dir="point_maze",
    ),
    "wall": dict(
        plan_config="plan_wall",
        gdrive_online="12BXoo7WYJbhy976Ee_e43NiQfHBk82tT",
        gdrive_adversarial="1fWdI-dA1HOsPGg-CdQZPrXkSw8LoDELO",
        dataset_dir="wall_single",
    ),
}

# Possible subdirectory names inside the extracted outputs.zip per environment
_PRETRAINED_SUBDIR_CANDIDATES = {
    "pusht":      ["pusht", "pusht_noise"],
    "point_maze": ["point_maze", "pointmaze", "maze"],
    "wall":       ["wall", "wall_single"],
}

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git", "wget", "unzip", "gcc", "g++",
        "libgl1-mesa-dev", "libgl1-mesa-glx", "libglew-dev",
        "libosmesa6-dev", "patchelf",
        "libglib2.0-0", "libsm6", "libxrender1", "libxext6",
        "libegl1", "libegl-mesa0", "libgles2",
        "libglvnd0", "libglvnd-dev", "libopengl0",
        "ffmpeg",
    )
    .run_commands(
        "wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco210.tar.gz",
        "mkdir -p /root/.mujoco && tar xzf /tmp/mujoco210.tar.gz -C /root/.mujoco",
    )
    .env({
        "LD_LIBRARY_PATH": "/root/.mujoco/mujoco210/bin",
        "MUJOCO_GL": "egl",
        "SDL_VIDEODRIVER": "dummy",
    })
    .pip_install("numpy==1.26.4", "Cython<3")
    .pip_install("mujoco-py==2.1.2.14")
    .pip_install(
        "torch==2.3.0",
        "torchvision==0.18.0",
        "hydra-core==1.2.0",
        "omegaconf==2.3.0",
        "hydra-submitit-launcher==1.2.0",
        "einops==0.4.1",
        "wandb",
        "gdown",
        "moviepy==1.0.3",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.4.9",
        "matplotlib",
        "scipy",
        "transformers==4.21.1",
        "gym==0.23.1",
        "pygame==2.5.2",
        "pymunk==6.8.0",
        "decord==0.6.0",
        "requests",
        "pillow",
        "accelerate==0.26.1",
        "submitit==1.5.1",
        "dm-control==1.0.27",
        "mujoco==3.2.7",
        "shapely",
        "opencv-python-headless",
        "scikit-image",
    )
    .pip_install("d4rl @ git+https://github.com/Farama-Foundation/d4rl.git@master#egg=d4rl")
    .run_commands("python -c 'import mujoco_py; mujoco_py.cymj'")
    .add_local_dir(
        ".",
        "/app",
        copy=True,
        ignore=[
            ".git",
            "**/__pycache__",
            "**/*.pyc",
            "outputs",
            "plan_outputs",
            "*.egg-info",
            ".venv",
        ],
    )
    .env({"PYTHONPATH": "/app"})
)

_CPU_KWARGS = dict(
    image=image,
    volumes={ROBUST_WM_HOME: volume},
    timeout=3600 * 4,
    cpu=4.0,
)

_GPU_PLAN_KWARGS = dict(
    image=image,
    volumes={ROBUST_WM_HOME: volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
    gpu="A100-40GB",
    timeout=3600 * 4,
    cpu=8.0,
)

_GPU_SWEEP_KWARGS = dict(
    image=image,
    volumes={ROBUST_WM_HOME: volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
    gpu="A100-40GB",
    timeout=3600 * 10,  # 10 h — CEM runs can exceed 5 h
    cpu=8.0,
)

_CPU_SWEEP_KWARGS = dict(
    image=image,
    volumes={ROBUST_WM_HOME: volume},
    timeout=3600 * 1,
    cpu=4.0,
)

# Sweep parameter values
_CEM_NUM_SAMPLES = [20, 50, 100, 200, 300]
_GD_OPT_STEPS = [5, 10, 20, 50, 100]


# ---------------------------------------------------------------------------
# Shared helpers (run inside containers, so no Modal decorators)
# ---------------------------------------------------------------------------

def _download_url(url: str, dest: str) -> None:
    import requests
    from pathlib import Path
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
    print(f"\nSaved to {dest} ({Path(dest).stat().st_size / 1e9:.2f} GB)")


def _extract_archive(archive_path: str, extract_dir: str) -> None:
    import zipfile
    import tarfile
    from pathlib import Path
    archive_path, extract_dir = Path(archive_path), Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    name = archive_path.name
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extract_dir)
    elif name.endswith(".tar.gz") or name.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(extract_dir)
    elif name.endswith(".tar"):
        with tarfile.open(archive_path, "r") as tf:
            tf.extractall(extract_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")
    print(f"Extracted to {extract_dir}")


def _find_checkpoint_root(base_dir):
    """Find the directory that has both hydra.yaml and checkpoints/ inside an extracted archive."""
    from pathlib import Path
    base_dir = Path(base_dir)
    # Breadth-first search up to 4 levels deep
    for hit in sorted(base_dir.rglob("hydra.yaml")):
        if (hit.parent / "checkpoints").is_dir():
            return hit.parent
    raise FileNotFoundError(f"No directory with hydra.yaml + checkpoints/ found under {base_dir}")


def _patch_hydra_yaml(hydra_yaml_path, datasets_root: str) -> None:
    """Replace data_path values in hydra.yaml so they point to the volume dataset directory."""
    import re
    from pathlib import Path
    hydra_yaml_path = Path(hydra_yaml_path)
    text = hydra_yaml_path.read_text()
    original = text

    DATASET_DIRS = ["pusht_noise", "point_maze", "wall_single"]

    def replacer(m):
        value = m.group(1).strip().strip("'\"")
        for dname in DATASET_DIRS:
            if dname in value:
                new_path = str(Path(datasets_root) / dname)
                return m.group(0).replace(value, new_path)
        return m.group(0)

    text = re.sub(r"data_path:\s*([^\n]+)", replacer, text)
    if text != original:
        hydra_yaml_path.write_text(text)
        print(f"Patched data_path in {hydra_yaml_path}")
    else:
        print(f"No data_path entries to patch in {hydra_yaml_path}")


# ---------------------------------------------------------------------------
# Download functions
# ---------------------------------------------------------------------------

@app.function(**_CPU_KWARGS)
def download_pretrained():
    """Download all pretrained DINO-WM checkpoints from OSF (one archive, all envs)."""
    import shutil
    from pathlib import Path

    datasets_root = str(Path(ROBUST_WM_HOME) / "datasets")
    cache_dir = Path(ROBUST_WM_HOME) / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    volume.reload()
    all_done = all(
        (Path(ROBUST_WM_HOME) / "checkpoints" / env / "pretrained" / "hydra.yaml").exists()
        for env in ENV_CONFIGS
    )
    if all_done:
        print("All pretrained checkpoints already on volume.")
        return

    zip_path = cache_dir / "outputs.zip"
    if not zip_path.exists():
        print("Downloading outputs.zip from OSF...")
        _download_url(OSF_OUTPUTS_URL, str(zip_path))
        volume.commit()
    else:
        print(f"outputs.zip already cached ({zip_path.stat().st_size / 1e9:.1f} GB).")

    extract_dir = Path("/tmp/outputs_extracted")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    _extract_archive(str(zip_path), str(extract_dir))

    hydra_files = list(extract_dir.rglob("hydra.yaml"))
    print(f"Found {len(hydra_files)} hydra.yaml files: {[str(h) for h in hydra_files]}")

    for env_name in ENV_CONFIGS:
        dst = Path(ROBUST_WM_HOME) / "checkpoints" / env_name / "pretrained"
        if (dst / "hydra.yaml").exists():
            print(f"[{env_name}/pretrained] Already on volume, skipping.")
            continue

        # Find the matching checkpoint root by searching for env-specific subdirectory names
        ckpt_root = None
        for candidate_name in _PRETRAINED_SUBDIR_CANDIDATES[env_name]:
            for hf in hydra_files:
                if candidate_name in str(hf) and (hf.parent / "checkpoints").is_dir():
                    ckpt_root = hf.parent
                    break
            if ckpt_root:
                break

        # Fallback: if archive has exactly one checkpoint root, use it for all envs
        if ckpt_root is None:
            roots = []
            for hf in hydra_files:
                if (hf.parent / "checkpoints").is_dir():
                    roots.append(hf.parent)
            if len(roots) == 1:
                ckpt_root = roots[0]
            elif len(roots) > 1:
                print(f"  Ambiguous — found {len(roots)} roots; picking first for {env_name}: {roots[0]}")
                ckpt_root = roots[0]

        if ckpt_root is None:
            print(f"WARNING: could not find pretrained ckpt for {env_name}. hydra.yaml files: {hydra_files}")
            continue

        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ckpt_root, dst, dirs_exist_ok=True)
        _patch_hydra_yaml(dst / "hydra.yaml", datasets_root)
        print(f"[{env_name}/pretrained] {ckpt_root} → {dst}")

    volume.commit()
    print("Pretrained checkpoints committed to volume.")


@app.function(**_CPU_KWARGS)
def download_checkpoints(env_name: str, method: str):
    """Download one online or adversarial checkpoint from Google Drive into the volume."""
    import gdown
    import shutil
    from pathlib import Path

    if env_name not in ENV_CONFIGS:
        raise ValueError(f"Unknown env_name={env_name!r}; choose from {list(ENV_CONFIGS)}")
    if method not in ("online", "adversarial"):
        raise ValueError(f"method must be 'online' or 'adversarial', got {method!r}")

    dst = Path(ROBUST_WM_HOME) / "checkpoints" / env_name / method
    volume.reload()
    if (dst / "hydra.yaml").exists():
        print(f"[{env_name}/{method}] Already on volume at {dst}.")
        return

    cfg = ENV_CONFIGS[env_name]
    gdrive_id = cfg[f"gdrive_{method}"]
    datasets_root = str(Path(ROBUST_WM_HOME) / "datasets")

    archive_path = f"/tmp/{env_name}_{method}_ckpt.zip"
    print(f"[{env_name}/{method}] Downloading from Google Drive (id={gdrive_id})...")
    downloaded = gdown.download(id=gdrive_id, output=archive_path, quiet=False)
    if downloaded is None:
        raise RuntimeError(f"gdown failed to download id={gdrive_id}")

    extract_dir = Path(f"/tmp/{env_name}_{method}_extracted")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    _extract_archive(downloaded, str(extract_dir))

    ckpt_root = _find_checkpoint_root(extract_dir)
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ckpt_root, dst, dirs_exist_ok=True)
    _patch_hydra_yaml(dst / "hydra.yaml", datasets_root)

    volume.commit()
    print(f"[{env_name}/{method}] Committed to volume at {dst}.")


@app.function(**_CPU_KWARGS)
def download_datasets(env_name: str):
    """Download and extract the training dataset for one environment into the volume."""
    import shutil
    from pathlib import Path

    if env_name not in ENV_CONFIGS:
        raise ValueError(f"Unknown env_name={env_name!r}; choose from {list(ENV_CONFIGS)}")

    cfg = ENV_CONFIGS[env_name]
    dataset_dir = cfg["dataset_dir"]
    dst = Path(ROBUST_WM_HOME) / "datasets" / dataset_dir

    volume.reload()
    if dst.exists() and any(dst.iterdir()):
        print(f"[{env_name}] Dataset already on volume at {dst}.")
        return

    zip_path = f"/tmp/{dataset_dir}.zip"
    print(f"[{env_name}] Downloading {dataset_dir}.zip from OSF...")
    _download_url(DATASET_URLS[env_name], zip_path)

    extract_dir = Path(f"/tmp/{dataset_dir}_extracted")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    _extract_archive(zip_path, str(extract_dir))

    # The zip may extract to a single named subdirectory — unwrap it
    contents = list(extract_dir.iterdir())
    src = contents[0] if (len(contents) == 1 and contents[0].is_dir()) else extract_dir

    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)

    volume.commit()
    print(f"[{env_name}] Dataset committed to volume at {dst}.")


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

@app.function(**_GPU_PLAN_KWARGS)
def plan(
    env_name: str = "pusht",
    method: str = "pretrained",
    seed: int = 42,
    n_evals: int = 50,
):
    """Run one seed of MPC planning and save results to the volume."""
    import os
    import subprocess
    from pathlib import Path

    if env_name not in ENV_CONFIGS:
        raise ValueError(f"Unknown env_name={env_name!r}; choose from {list(ENV_CONFIGS)}")

    cfg = ENV_CONFIGS[env_name]
    ckpt_path = str(Path(ROBUST_WM_HOME) / "checkpoints" / env_name / method)
    output_dir = f"{ROBUST_WM_HOME}/plan_outputs/{env_name}/{method}/seed{seed}"

    env = {**os.environ, "MUJOCO_GL": "egl", "SDL_VIDEODRIVER": "dummy", "PYTHONPATH": "/app"}

    cmd = [
        "python", "plan.py",
        "--config-name", cfg["plan_config"],
        f"ckpt_path={ckpt_path}",
        f"n_evals={n_evals}",
        f"seed={seed}",
        f"hydra.run.dir={output_dir}",
    ]
    print(f"[{env_name}/{method}/seed={seed}] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd="/app", env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Planning failed for {env_name}/{method}/seed={seed}")

    volume.commit()
    return {"env": env_name, "method": method, "seed": seed}


# ---------------------------------------------------------------------------
# Sweep planning
# ---------------------------------------------------------------------------

@app.function(**_GPU_SWEEP_KWARGS)
def plan_sweep(
    env_name: str = "pusht",
    method: str = "pretrained",
    planner_type: str = "gd",   # "gd" or "cem"
    sweep_value: int = 100,
    seed: int = 42,
    n_evals: int = 50,
):
    """Run one sweep cell: MPC+GD or MPC+CEM with a specific budget value."""
    import os
    import subprocess
    from pathlib import Path

    if env_name not in ENV_CONFIGS:
        raise ValueError(f"Unknown env_name={env_name!r}; choose from {list(ENV_CONFIGS)}")
    if planner_type not in ("gd", "cem"):
        raise ValueError(f"planner_type must be 'gd' or 'cem', got {planner_type!r}")

    ckpt_path = str(Path(ROBUST_WM_HOME) / "checkpoints" / env_name / method)
    output_dir = (
        f"{ROBUST_WM_HOME}/plan_outputs/sweep/{env_name}/{method}"
        f"/{planner_type}_{sweep_value}/seed{seed}"
    )

    # Auto-detect model epoch: prefer 'latest', fall back to highest numbered checkpoint
    ckpt_dir = Path(ckpt_path) / "checkpoints"
    volume.reload()
    if (ckpt_dir / "model_latest.pth").exists():
        model_epoch = "latest"
    else:
        pths = sorted(ckpt_dir.glob("model_*.pth"))
        if not pths:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
        # pick the highest epoch number
        model_epoch = pths[-1].stem.replace("model_", "")

    sweep_override = (
        f"planner.sub_planner.opt_steps={sweep_value}"
        if planner_type == "gd"
        else f"planner.sub_planner.num_samples={sweep_value}"
    )

    env = {**os.environ, "MUJOCO_GL": "egl", "SDL_VIDEODRIVER": "dummy", "PYTHONPATH": "/app"}

    cmd = [
        "python", "plan.py",
        "--config-name", f"plan_pusht_sweep_{planner_type}",
        f"ckpt_path={ckpt_path}",
        f"model_epoch={model_epoch}",
        f"n_evals={n_evals}",
        f"seed={seed}",
        f"hydra.run.dir={output_dir}",
        sweep_override,
    ]
    print(f"[{env_name}/{method}/{planner_type}/val={sweep_value}] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd="/app", env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"Sweep planning failed: {env_name}/{method}/{planner_type}/{sweep_value}"
        )

    volume.commit()
    return {
        "env": env_name,
        "method": method,
        "planner_type": planner_type,
        "sweep_value": sweep_value,
        "seed": seed,
        "output_dir": output_dir,
    }


@app.function(**_CPU_SWEEP_KWARGS)
def collect_and_plot():
    """Read logs.json files from the volume, write CSV, and generate comparison plot."""
    import csv
    import json
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    volume.reload()

    sweep_root = Path(ROBUST_WM_HOME) / "plan_outputs" / "sweep" / "pusht"
    results_dir = Path(ROBUST_WM_HOME) / "sweep_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Each config: (label, method, planner_type, sweep_values)
    configs = [
        ("pretrained/MPC+CEM", "pretrained", "cem", _CEM_NUM_SAMPLES),
        ("pretrained/MPC+GD",  "pretrained", "gd",  _GD_OPT_STEPS),
        ("adversarial/MPC+GD", "adversarial", "gd", _GD_OPT_STEPS),
    ]

    def read_run(method, planner_type, sweep_value, seed=42):
        run_dir = sweep_root / method / f"{planner_type}_{sweep_value}" / f"seed{seed}"
        logs_path = run_dir / "logs.json"
        if not logs_path.exists():
            print(f"  MISSING: {logs_path}")
            return None
        entries = []
        with open(logs_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        # Final success rate: last mpc/success_rate logged (step == max_iter)
        success_rate = None
        step_times = []
        for e in entries:
            if "mpc/success_rate" in e:
                success_rate = e["mpc/success_rate"]
            if "mpc/step_time_sec" in e:
                step_times.append(e["mpc/step_time_sec"])
        mean_step_time = float(np.mean(step_times)) if step_times else float("nan")
        return {"success_rate": success_rate, "mean_step_time_sec": mean_step_time}

    # Collect all results
    rows = []
    for label, method, planner_type, sweep_values in configs:
        for sv in sweep_values:
            r = read_run(method, planner_type, sv)
            rows.append({
                "config": label,
                "method": method,
                "planner_type": planner_type,
                "sweep_value": sv,
                "final_success_rate": r["success_rate"] if r else "",
                "mean_step_time_sec": r["mean_step_time_sec"] if r else "",
            })

    csv_path = results_dir / "pusht_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["config", "method", "planner_type", "sweep_value",
                           "final_success_rate", "mean_step_time_sec"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV written to {csv_path}")

    # --- Plot ---
    _, ax_bottom = plt.subplots(figsize=(8, 5))
    ax_top = ax_bottom.twiny()

    colors = {"pretrained/MPC+CEM": "tab:blue", "pretrained/MPC+GD": "tab:orange", "adversarial/MPC+GD": "tab:green"}
    markers = {"pretrained/MPC+CEM": "o", "pretrained/MPC+GD": "s", "adversarial/MPC+GD": "^"}
    x_positions = list(range(5))  # 0..4

    for label, method, planner_type, sweep_values in configs:
        ys = []
        xs = []
        for idx, sv in enumerate(sweep_values):
            r = read_run(method, planner_type, sv)
            if r and r["success_rate"] is not None:
                xs.append(idx)
                ys.append(r["success_rate"])
        if xs:
            ax_bottom.plot(
                xs, ys,
                color=colors[label], marker=markers[label],
                linewidth=2, markersize=7, label=label,
            )

    # Bottom axis: GD opt_steps
    ax_bottom.set_xticks(x_positions)
    ax_bottom.set_xticklabels([str(v) for v in _GD_OPT_STEPS])
    ax_bottom.set_xlabel("GD opt_steps  (MPC+GD lines)")
    ax_bottom.set_ylabel("Final success rate (after 10 MPC steps)")
    ax_bottom.set_ylim(0, 1.05)
    ax_bottom.set_xlim(-0.3, 4.3)

    # Top axis: CEM num_samples
    ax_top.set_xlim(ax_bottom.get_xlim())
    ax_top.set_xticks(x_positions)
    ax_top.set_xticklabels([str(v) for v in _CEM_NUM_SAMPLES])
    ax_top.set_xlabel("CEM num_samples  (MPC+CEM line)")

    ax_bottom.legend(loc="lower right")
    ax_bottom.grid(True, alpha=0.3)
    plt.title("PushT: planning success vs. computational budget (10 MPC steps)")
    plt.tight_layout()

    plot_path = results_dir / "pusht_sweep_plot.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")

    volume.commit()
    return str(plot_path)


# ---------------------------------------------------------------------------
# Remote orchestrators (run inside Modal — survive local disconnect)
# ---------------------------------------------------------------------------

@app.function(**{**_CPU_SWEEP_KWARGS, "timeout": 3600 * 12})
def _orchestrate_cem(n_evals: int = 50, seed: int = 42, values=None):
    """Runs inside a Modal container: waits for CEM jobs then plots."""
    if values is None:
        values = _CEM_NUM_SAMPLES
    handles = []
    for ns in values:
        h = plan_sweep.spawn(env_name="pusht", method="pretrained",
                             planner_type="cem", sweep_value=ns,
                             seed=seed, n_evals=n_evals)
        handles.append((f"cem/ns={ns}", h))
        print(f"  spawned pretrained/cem/ns={ns}")

    for tag, h in handles:
        try:
            ret = h.get()
            print(f"  DONE {tag} → {ret['output_dir']}")
        except Exception as e:
            print(f"  FAILED {tag}: {e}")

    print("CEM jobs done. Generating plot...")
    collect_and_plot.remote()
    print("Plot committed to volume.")


# ---------------------------------------------------------------------------
# Entrypoints
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def run_cem_sweep(n_evals: int = 50, seed: int = 42):
    """Launch CEM sweep jobs via a remote orchestrator (survives local disconnect).

    The orchestrator runs inside Modal — safe to close your terminal.
    Monitor progress via W&B project: pusht_sweep

    Examples:
        modal run modal_plan.py::run_cem_sweep
        modal run modal_plan.py::run_cem_sweep --n-evals 50 --seed 42
    """
    _orchestrate_cem.spawn(n_evals=n_evals, seed=seed)
    print("CEM orchestrator launched on Modal (runs independently).")
    print("Monitor: https://wandb.ai/avp2145-columbia-university/pusht_sweep")
    print("When done, download results:")
    print("  modal volume get stableworldmodel sweep_results/pusht_sweep_plot.png .")
    print("  modal volume get stableworldmodel sweep_results/pusht_sweep.csv .")


@app.local_entrypoint()
def run_one_sweep(
    env_name: str = "pusht",
    method: str = "adversarial",
    planner_type: str = "gd",
    sweep_value: int = 100,
    seed: int = 99,
    n_evals: int = 50,
):
    """Run a single sweep cell and print the result.

    Examples:
        modal run modal_plan.py::run_one_sweep
        modal run modal_plan.py::run_one_sweep --method adversarial --planner-type gd --sweep-value 100 --seed 99
    """
    ret = plan_sweep.remote(
        env_name=env_name,
        method=method,
        planner_type=planner_type,
        sweep_value=sweep_value,
        seed=seed,
        n_evals=n_evals,
    )
    print(f"Done. output_dir={ret['output_dir']}")
    print("To fetch logs:")
    print(f"  modal volume get stableworldmodel {ret['output_dir'].replace('/root/.robust_worldmodel/', 'sweep_results/')}logs.json .")


@app.local_entrypoint()
def run_sweep(n_evals: int = 50, seed: int = 42):
    """Spawn all 15 sweep jobs in parallel, report progress, then collect results.

    Examples:
        modal run modal_plan.py::run_sweep
        modal run modal_plan.py::run_sweep --n-evals 50 --seed 42
    """
    sweep_jobs = []
    # pretrained + MPC+CEM
    for ns in _CEM_NUM_SAMPLES:
        tag = f"pretrained/cem/ns={ns}"
        h = plan_sweep.spawn(env_name="pusht", method="pretrained",
                             planner_type="cem", sweep_value=ns,
                             seed=seed, n_evals=n_evals)
        sweep_jobs.append((tag, h))
        print(f"  spawned {tag}")

    # pretrained + MPC+GD
    for os_ in _GD_OPT_STEPS:
        tag = f"pretrained/gd/opt={os_}"
        h = plan_sweep.spawn(env_name="pusht", method="pretrained",
                             planner_type="gd", sweep_value=os_,
                             seed=seed, n_evals=n_evals)
        sweep_jobs.append((tag, h))
        print(f"  spawned {tag}")

    # adversarial + MPC+GD
    for os_ in _GD_OPT_STEPS:
        tag = f"adversarial/gd/opt={os_}"
        h = plan_sweep.spawn(env_name="pusht", method="adversarial",
                             planner_type="gd", sweep_value=os_,
                             seed=seed, n_evals=n_evals)
        sweep_jobs.append((tag, h))
        print(f"  spawned {tag}")

    print(f"\nAll {len(sweep_jobs)} jobs spawned. Waiting for results...\n")
    for tag, h in sweep_jobs:
        try:
            ret = h.get()
            print(f"  DONE  {tag}  →  output_dir={ret['output_dir']}")
        except Exception as e:
            print(f"  FAILED {tag}: {e}")

    print("\nAll sweep jobs finished. Generating plot...")
    plot_path = collect_and_plot.remote()
    print(f"Plot saved at: {plot_path}")
    print(f"Download with:  modal volume get stableworldmodel sweep_results/pusht_sweep_plot.png .")
    print(f"CSV download:   modal volume get stableworldmodel sweep_results/pusht_sweep.csv .")


@app.local_entrypoint()
def download_all():
    """Spawn all checkpoint + dataset downloads in parallel, then wait for completion."""
    print("Spawning all downloads in parallel...")

    h_pretrained = download_pretrained.spawn()

    gdrive_handles = [
        (env_name, method, download_checkpoints.spawn(env_name=env_name, method=method))
        for env_name in ENV_CONFIGS
        for method in ("online", "adversarial")
    ]
    dataset_handles = [
        (env_name, download_datasets.spawn(env_name=env_name))
        for env_name in ENV_CONFIGS
    ]

    print("Waiting for pretrained checkpoints (OSF outputs.zip ~953 MB)...")
    h_pretrained.get()
    print("  pretrained done.")

    for env_name, method, h in gdrive_handles:
        h.get()
        print(f"  [{env_name}/{method}] done.")

    for env_name, h in dataset_handles:
        h.get()
        print(f"  [{env_name}] dataset done.")

    print("All downloads complete.")


@app.local_entrypoint()
def main(
    env_name: str = "pusht",
    method: str = "pretrained",
    seed: int = 42,
    n_evals: int = 50,
):
    """Run one seed of MPC planning.

    Examples:
        modal run modal_plan.py --env-name pusht --method pretrained
        modal run modal_plan.py --env-name wall --method adversarial
    """
    ret = plan.remote(
        env_name=env_name,
        method=method,
        seed=seed,
        n_evals=n_evals,
    )
    print(f"Planning result: {ret}")


@app.local_entrypoint()
def plot():
    """Collect results from volume and regenerate the sweep plot/CSV.

    Run after all sweep jobs have finished:
        modal run modal_plan.py::plot
    Then download:
        modal volume get stableworldmodel sweep_results/pusht_sweep_plot.png .
        modal volume get stableworldmodel sweep_results/pusht_sweep.csv .
    """
    plot_path = collect_and_plot.remote()
    print(f"Plot saved at: {plot_path}")
    print("Download with:")
    print("  modal volume get stableworldmodel sweep_results/pusht_sweep_plot.png .")
    print("  modal volume get stableworldmodel sweep_results/pusht_sweep.csv .")

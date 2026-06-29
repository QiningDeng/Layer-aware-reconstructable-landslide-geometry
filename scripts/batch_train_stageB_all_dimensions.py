#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sequential batch launcher for the two Stage B latent models.

Default training plan
---------------------
Model order:
    1. VAE
    2. Joint implicit auto-decoder

Latent dimensions:
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512

Training controls:
    target maximum epoch = 5000
    early-stopping patience = 40
    scheduler = none
    validation after training = enabled
    test evaluation after training = disabled

Each run is executed in an independent Python subprocess. This releases GPU
memory between runs and prevents one failed run from corrupting the next run.

Expected output structure
-------------------------
<output_root>/
    stageB_vae--latent_dim_1/
    stageB_vae--latent_dim_2/
    ...
    stageB_vae--latent_dim_512/
    stageB_autodecoder--latent_dim_1/
    ...
    stageB_autodecoder--latent_dim_512/
    _batch_training_logs/
        batch_YYYYMMDD_HHMMSS/
            batch_config.json
            batch_training_plan.csv
            batch_status.csv
            current_run.txt
            logs/
                001_vae_latent_dim_1.log
                ...

Existing-run policies
---------------------
archive   : rename an existing run folder before starting a fresh run
overwrite : delete an existing run folder before starting
skip      : leave an existing run folder untouched and skip that run
error     : stop before training if an expected run folder already exists

Recommended usage
-----------------
Place this launcher beside:
    stageB_joint_implicit_train_eval_predict_timing_resume.py

Then run:
    python batch_train_stageB_all_dimensions.py

The launcher will ask for:
1. the Stage A feature-database root;
2. the Stage B output root;
3. the training script, only if it cannot be found automatically.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:
    tk = None
    filedialog = None


DEFAULT_MODELS = ["vae", "autodecoder"]
DEFAULT_LATENT_DIMS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
DEFAULT_TRAINER_NAME = "stageB_joint_implicit_train_eval_predict_timing_resume_realtime.py"


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def timestamp_compact() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, remainder = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def select_folder(title: str) -> str:
    if tk is None or filedialog is None:
        raise RuntimeError(
            "tkinter is unavailable. Please specify the path on the command line."
        )
    root = tk.Tk()
    root.withdraw()
    root.update()
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    return folder


def select_file(title: str) -> str:
    if tk is None or filedialog is None:
        raise RuntimeError(
            "tkinter is unavailable. Please specify --trainer_script."
        )
    root = tk.Tk()
    root.withdraw()
    root.update()
    path = filedialog.askopenfilename(
        title=title,
        filetypes=[("Python script", "*.py"), ("All files", "*.*")],
    )
    root.destroy()
    return path


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_folder_name(model_type: str, latent_dim: int) -> str:
    return f"stageB_{model_type}--latent_dim_{int(latent_dim)}"


def unique_archive_path(path: Path) -> Path:
    base = path.with_name(f"{path.name}__archive_{timestamp_compact()}")
    candidate = base
    index = 1
    while candidate.exists():
        candidate = path.with_name(f"{base.name}_{index:02d}")
        index += 1
    return candidate


def apply_existing_run_policy(
    run_dir: Path,
    policy: str,
    dry_run: bool,
) -> Tuple[str, str]:
    """
    Returns:
        action: train / skip
        note
    """
    if not run_dir.exists():
        return "train", "No existing run folder."

    if policy == "skip":
        return "skip", "Existing run folder retained; run skipped."

    if policy == "error":
        raise FileExistsError(
            f"Expected output folder already exists: {run_dir}\n"
            "Choose a new output root or use --existing_run_policy "
            "archive, overwrite, or skip."
        )

    if policy == "overwrite":
        if not dry_run:
            shutil.rmtree(run_dir)
        return "train", "Existing run folder removed before fresh training."

    if policy == "archive":
        archive_path = unique_archive_path(run_dir)
        if not dry_run:
            run_dir.rename(archive_path)
        return "train", f"Existing run folder archived as: {archive_path.name}"

    raise ValueError(f"Unsupported existing-run policy: {policy}")


def build_training_plan(
    models: Sequence[str],
    latent_dims: Sequence[int],
    order: str,
) -> List[Tuple[str, int]]:
    models = list(models)
    latent_dims = [int(v) for v in latent_dims]

    if order == "model_major":
        return [(model, dim) for model in models for dim in latent_dims]

    if order == "dimension_major":
        return [(model, dim) for dim in latent_dims for model in models]

    raise ValueError(f"Unsupported batch order: {order}")


def resolve_trainer_script(user_path: str) -> Path:
    if user_path:
        path = Path(user_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    launcher_dir = Path(__file__).resolve().parent
    candidate = launcher_dir / DEFAULT_TRAINER_NAME
    if candidate.is_file():
        return candidate

    print(
        f"The training script '{DEFAULT_TRAINER_NAME}' was not found beside "
        "the batch launcher. Please select it."
    )
    selected = select_file("Select the Stage B training script")
    if not selected:
        raise ValueError("No training script was selected.")
    path = Path(selected).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def build_child_command(
    python_executable: str,
    trainer_script: Path,
    feature_root: Path,
    output_root: Path,
    model_type: str,
    latent_dim: int,
    args: argparse.Namespace,
) -> List[str]:
    command = [
        python_executable,
        "-u",
        str(trainer_script),
        "train",
        "--feature_root",
        str(feature_root),
        "--output_root",
        str(output_root),
        "--model_type",
        model_type,
        "--latent_dim",
        str(int(latent_dim)),
        "--encoder_input_size",
        str(int(args.encoder_input_size)),
        "--eval_grid_size",
        str(int(args.eval_grid_size)),
        "--batch_size",
        str(int(args.batch_size)),
        "--epochs",
        str(int(args.epochs)),
        "--learning_rate",
        str(float(args.learning_rate)),
        "--latent_learning_rate",
        str(float(args.latent_learning_rate)),
        "--points_per_sample",
        str(int(args.points_per_sample)),
        "--seed",
        str(int(args.seed)),
        "--num_workers",
        str(int(args.num_workers)),
        "--patience",
        str(int(args.patience)),
        "--save_every",
        str(int(args.save_every)),
        "--plot_every",
        str(int(args.plot_every)),
        "--scheduler_type",
        args.scheduler_type,
        "--scheduler_factor",
        str(float(args.scheduler_factor)),
        "--scheduler_patience",
        str(int(args.scheduler_patience)),
        "--min_learning_rate",
        str(float(args.min_learning_rate)),
    ]

    if args.cache_in_memory:
        command.append("--cache_in_memory")

    if args.no_final_evaluation:
        command.append("--no_final_evaluation")

    if args.evaluate_test_after_training:
        command.append("--evaluate_test_after_training")

    return command


def stream_subprocess(
    command: Sequence[str],
    log_path: Path,
    env: Dict[str, str],
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        log_file.write("Command:\n")
        log_file.write(subprocess.list2cmdline(list(command)))
        log_file.write("\n\n")
        log_file.flush()

        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )

        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
            return int(process.wait())
        except KeyboardInterrupt:
            print("\nBatch launcher received Ctrl+C. Terminating the current run...")
            log_file.write("\n[Batch launcher interrupted by Ctrl+C]\n")
            log_file.flush()
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially train the VAE and auto-decoder for all requested "
            "latent dimensions."
        )
    )

    parser.add_argument("--trainer_script", type=str, default="")
    parser.add_argument("--feature_root", type=str, default="")
    parser.add_argument("--output_root", type=str, default="")

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["vae", "autodecoder"],
        default=DEFAULT_MODELS,
    )
    parser.add_argument(
        "--latent_dims",
        nargs="+",
        type=int,
        default=DEFAULT_LATENT_DIMS,
    )
    parser.add_argument(
        "--batch_order",
        choices=["model_major", "dimension_major"],
        default="model_major",
        help=(
            "model_major: VAE dimensions first, then auto-decoder dimensions. "
            "dimension_major: train both models at each dimension before moving "
            "to the next dimension."
        ),
    )

    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--encoder_input_size", type=int, default=128)
    parser.add_argument("--eval_grid_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--latent_learning_rate", type=float, default=5e-3)
    parser.add_argument("--points_per_sample", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--plot_every", type=int, default=10)
    parser.add_argument(
        "--scheduler_type",
        choices=["none", "plateau"],
        default="none",
    )
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--scheduler_patience", type=int, default=10)
    parser.add_argument("--min_learning_rate", type=float, default=1e-7)
    parser.add_argument("--cache_in_memory", action="store_true")

    parser.add_argument(
        "--existing_run_policy",
        choices=["archive", "overwrite", "skip", "error"],
        default="archive",
        help=(
            "Default 'archive' preserves old run folders and starts fresh runs "
            "with the standard folder names."
        ),
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue with the next run when one child process fails.",
    )
    parser.add_argument(
        "--no_final_evaluation",
        action="store_true",
        help="Skip the automatic validation evaluation after each run.",
    )
    parser.add_argument(
        "--evaluate_test_after_training",
        action="store_true",
        help=(
            "Also evaluate the test split after every run. This is discouraged "
            "during latent-dimension selection."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Create the training plan and print commands without launching training.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    feature_root_text = args.feature_root.strip()
    if not feature_root_text:
        print("Please select the Stage A feature-database root folder.")
        feature_root_text = select_folder(
            "Select the Stage A feature-database root folder"
        )
    if not feature_root_text:
        raise ValueError("No Stage A feature root was selected.")
    feature_root = Path(feature_root_text).expanduser().resolve()
    if not feature_root.is_dir():
        raise NotADirectoryError(feature_root)

    output_root_text = args.output_root.strip()
    if not output_root_text:
        print("Please select the Stage B output root folder.")
        output_root_text = select_folder(
            "Select the Stage B output root folder"
        )
    if not output_root_text:
        raise ValueError("No Stage B output root was selected.")
    output_root = Path(output_root_text).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    trainer_script = resolve_trainer_script(args.trainer_script)
    python_executable = str(Path(sys.executable).resolve())

    plan = build_training_plan(
        args.models,
        args.latent_dims,
        args.batch_order,
    )

    batch_id = f"batch_{timestamp_compact()}"
    batch_dir = output_root / "_batch_training_logs" / batch_id
    run_logs_dir = batch_dir / "logs"
    batch_dir.mkdir(parents=True, exist_ok=False)
    run_logs_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "batch_id": batch_id,
        "created_at": now_iso(),
        "python_executable": python_executable,
        "trainer_script": str(trainer_script),
        "feature_root": str(feature_root),
        "output_root": str(output_root),
        "models": list(args.models),
        "latent_dims": [int(v) for v in args.latent_dims],
        "batch_order": args.batch_order,
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "encoder_input_size": int(args.encoder_input_size),
        "eval_grid_size": int(args.eval_grid_size),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "latent_learning_rate": float(args.latent_learning_rate),
        "points_per_sample": int(args.points_per_sample),
        "seed": int(args.seed),
        "num_workers": int(args.num_workers),
        "save_every": int(args.save_every),
        "plot_every": int(args.plot_every),
        "scheduler_type": args.scheduler_type,
        "existing_run_policy": args.existing_run_policy,
        "continue_on_error": bool(args.continue_on_error),
        "final_validation_evaluation": not args.no_final_evaluation,
        "test_evaluation_after_training": bool(
            args.evaluate_test_after_training
        ),
        "dry_run": bool(args.dry_run),
    }
    write_json(batch_dir / "batch_config.json", config_payload)

    plan_rows: List[Dict[str, Any]] = []
    for sequence, (model_type, latent_dim) in enumerate(plan, start=1):
        run_name = run_folder_name(model_type, latent_dim)
        command = build_child_command(
            python_executable,
            trainer_script,
            feature_root,
            output_root,
            model_type,
            latent_dim,
            args,
        )
        plan_rows.append({
            "sequence": sequence,
            "model_type": model_type,
            "latent_dim": latent_dim,
            "run_name": run_name,
            "run_directory": str(output_root / run_name),
            "command": subprocess.list2cmdline(command),
        })

    plan_fields = [
        "sequence",
        "model_type",
        "latent_dim",
        "run_name",
        "run_directory",
        "command",
    ]
    write_csv(
        batch_dir / "batch_training_plan.csv",
        plan_rows,
        plan_fields,
    )

    print("")
    print("=" * 78)
    print("Stage B sequential batch training")
    print("=" * 78)
    print(f"Trainer: {trainer_script}")
    print(f"Feature root: {feature_root}")
    print(f"Output root: {output_root}")
    print(f"Runs: {len(plan)}")
    print(f"Target maximum epoch: {args.epochs}")
    print(f"Early-stopping patience: {args.patience}")
    print(f"Batch order: {args.batch_order}")
    print(f"Existing-run policy: {args.existing_run_policy}")
    print(f"Batch log directory: {batch_dir}")
    print("Real-time per-epoch console output: enabled (unbuffered child process)")
    print("Test evaluation after each run: "
          + ("enabled" if args.evaluate_test_after_training else "disabled"))
    print("=" * 78)

    status_fields = [
        "sequence",
        "model_type",
        "latent_dim",
        "run_name",
        "status",
        "started_at",
        "finished_at",
        "elapsed_seconds",
        "elapsed_hhmmss",
        "return_code",
        "existing_run_action",
        "note",
        "log_file",
    ]
    statuses: List[Dict[str, Any]] = []

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    batch_start = time.perf_counter()

    try:
        for sequence, (model_type, latent_dim) in enumerate(plan, start=1):
            run_name = run_folder_name(model_type, latent_dim)
            run_dir = output_root / run_name
            log_path = (
                run_logs_dir
                / f"{sequence:03d}_{model_type}_latent_dim_{latent_dim}.log"
            )

            action, policy_note = apply_existing_run_policy(
                run_dir,
                args.existing_run_policy,
                args.dry_run,
            )

            current_payload = {
                "sequence": sequence,
                "total_runs": len(plan),
                "model_type": model_type,
                "latent_dim": latent_dim,
                "run_name": run_name,
                "updated_at": now_iso(),
                "status": "planned" if args.dry_run else action,
            }
            write_json(batch_dir / "current_run.json", current_payload)
            (batch_dir / "current_run.txt").write_text(
                (
                    f"{sequence}/{len(plan)} | "
                    f"{model_type} | latent_dim={latent_dim} | {action}\n"
                ),
                encoding="utf-8",
            )

            if action == "skip":
                status = {
                    "sequence": sequence,
                    "model_type": model_type,
                    "latent_dim": latent_dim,
                    "run_name": run_name,
                    "status": "skipped",
                    "started_at": "",
                    "finished_at": now_iso(),
                    "elapsed_seconds": 0.0,
                    "elapsed_hhmmss": "00:00:00",
                    "return_code": "",
                    "existing_run_action": args.existing_run_policy,
                    "note": policy_note,
                    "log_file": str(log_path),
                }
                statuses.append(status)
                write_csv(
                    batch_dir / "batch_status.csv",
                    statuses,
                    status_fields,
                )
                print(
                    f"[{sequence}/{len(plan)}] Skipped "
                    f"{model_type}, latent_dim={latent_dim}: {policy_note}"
                )
                continue

            command = build_child_command(
                python_executable,
                trainer_script,
                feature_root,
                output_root,
                model_type,
                latent_dim,
                args,
            )

            print("")
            print("-" * 78)
            print(
                f"[{sequence}/{len(plan)}] Starting "
                f"{model_type}, latent_dim={latent_dim}"
            )
            print(f"Existing-run handling: {policy_note}")
            print(f"Log file: {log_path}")
            print("-" * 78)

            started_at = now_iso()
            run_start = time.perf_counter()

            if args.dry_run:
                print(subprocess.list2cmdline(command))
                return_code = 0
                run_status = "dry_run"
            else:
                return_code = stream_subprocess(command, log_path, env)
                run_status = "completed" if return_code == 0 else "failed"

            elapsed = time.perf_counter() - run_start
            finished_at = now_iso()

            status = {
                "sequence": sequence,
                "model_type": model_type,
                "latent_dim": latent_dim,
                "run_name": run_name,
                "status": run_status,
                "started_at": started_at,
                "finished_at": finished_at,
                "elapsed_seconds": float(elapsed),
                "elapsed_hhmmss": format_seconds(elapsed),
                "return_code": int(return_code),
                "existing_run_action": args.existing_run_policy,
                "note": policy_note,
                "log_file": str(log_path),
            }
            statuses.append(status)
            write_csv(
                batch_dir / "batch_status.csv",
                statuses,
                status_fields,
            )

            print(
                f"[{sequence}/{len(plan)}] {run_status}: "
                f"{model_type}, latent_dim={latent_dim}, "
                f"elapsed={format_seconds(elapsed)}"
            )

            if return_code != 0 and not args.continue_on_error:
                raise RuntimeError(
                    f"Training failed for {run_name} with return code "
                    f"{return_code}. See: {log_path}"
                )

    except KeyboardInterrupt:
        write_json(
            batch_dir / "batch_interrupted.json",
            {
                "interrupted_at": now_iso(),
                "completed_status_rows": len(statuses),
                "total_planned_runs": len(plan),
            },
        )
        print(f"\nBatch training was interrupted. Logs are preserved in: {batch_dir}")
        raise
    finally:
        batch_elapsed = time.perf_counter() - batch_start
        write_json(
            batch_dir / "batch_summary.json",
            {
                "batch_id": batch_id,
                "finished_at": now_iso(),
                "planned_runs": len(plan),
                "status_rows": len(statuses),
                "completed_runs": sum(
                    1 for row in statuses if row["status"] == "completed"
                ),
                "failed_runs": sum(
                    1 for row in statuses if row["status"] == "failed"
                ),
                "skipped_runs": sum(
                    1 for row in statuses if row["status"] == "skipped"
                ),
                "dry_runs": sum(
                    1 for row in statuses if row["status"] == "dry_run"
                ),
                "batch_elapsed_seconds": float(batch_elapsed),
                "batch_elapsed_hhmmss": format_seconds(batch_elapsed),
            },
        )

    print("")
    print("=" * 78)
    print("Sequential batch training finished.")
    print(f"Batch elapsed time: {format_seconds(time.perf_counter() - batch_start)}")
    print(f"Batch logs: {batch_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()

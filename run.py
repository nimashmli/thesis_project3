import argparse
import json
import os
import numpy as np

from kfold_validation import validate
from model_use.main import choose_model
from plot import plot_training_history, plot_subject_dependet
from run_utils import build_run_dir, ensure_dir, save_json, status_path, load_json


def parse_args():
    parser = argparse.ArgumentParser(description="Unified runner for EEG models.")
    parser.add_argument("--config", help="Path to JSON config file.", default=None)
    parser.add_argument("--results_root", default="results", help="Root directory for outputs.")
    parser.add_argument("--run_dir", help="Existing run directory to resume.")
    parser.add_argument(
        "--mode",
        choices=["subject_independent", "subject_dependent"],
        help="Training/validation mode.",
    )
    parser.add_argument("--model", help="Model name (e.g., simpleNN).")
    parser.add_argument("--emotion", help="Emotion label to filter data.")
    parser.add_argument(
        "--category",
        choices=["binary", "5category"],
        help="Labeling strategy.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of folds (subject-dependent k-fold or outer k for validation).",
    )
    parser.add_argument(
        "--num_people",
        type=int,
        default=23,
        help="Total number of subjects for subject-independent validation.",
    )
    args = parser.parse_args()

    # اگر run_dir داده شده، ابتدا config.json از آن پوشه را بخوان
    if args.run_dir:
        run_config_path = os.path.join(args.run_dir, "config.json")
        if os.path.exists(run_config_path):
            with open(run_config_path, "r", encoding="utf-8") as f:
                run_cfg = json.load(f)
            for key, val in run_cfg.items():
                if hasattr(args, key):
                    cur = getattr(args, key)
                    if cur is None or cur == parser.get_default(key):
                        setattr(args, key, val)

    # load config if provided (این می‌تواند override کند)
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for key, val in cfg.items():
            if hasattr(args, key):
                cur = getattr(args, key)
                if cur is None or cur == parser.get_default(key):
                    setattr(args, key, val)

    required = ["mode", "model", "emotion", "category"]
    missing = [r for r in required if getattr(args, r) is None]
    if missing:
        raise ValueError(f"Missing required args: {missing}")
    return args


def run_subject_independent(args, run_dir):
    status_file = status_path(run_dir)
    save_json(
        status_file,
        {
            "mode": "subject_independent",
            "status": "running",
            "current_epoch": 0,
            "total_epochs": None,
        },
    )

    train_loss, val_loss, train_acc, val_acc = validate(
        args.model, args.emotion, args.category, args.k, args.num_people, run_dir=run_dir
    )
    history = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
    }
    save_json(os.path.join(run_dir, "history.json"), history)
    save_json(
        status_file,
        {
            "mode": "subject_independent",
            "status": "completed",
            "current_epoch": len(train_loss),
            "total_epochs": len(train_loss),
        },
    )
    print(
        f"\nSubject-independent results (averaged over {args.k} folds):\n"
        f"  Train Loss: {np.mean(train_loss[-5:]):.4f}\n"
        f"  Val   Loss: {np.mean(val_loss[-5:]):.4f}\n"
        f"  Train Acc : {np.mean(train_acc[-5:]):.2f}%\n"
        f"  Val   Acc : {np.mean(val_acc[-5:]):.2f}%\n"
    )
    plot_training_history(history)


def run_subject_dependent(args, run_dir):
    status_file = status_path(run_dir)
    save_json(
        status_file,
        {
            "mode": "subject_dependent",
            "status": "running",
            "current_subject": 0,
            "total_subjects": None,
            "current_fold": 0,
            "total_folds": args.k,
        },
    )

    accuracies = choose_model(
        args.model,
        args.emotion,
        args.category,
        test_person=None,
        fold_idx=None,
        subject_dependecy="subject_dependent",
        k=args.k,
        run_dir=run_dir,
    )
    save_json(os.path.join(run_dir, "history.json"), accuracies)
    save_json(
        status_file,
        {
            "mode": "subject_dependent",
            "status": "completed",
            "current_subject": None,
            "total_subjects": len(accuracies.get("test", [])),
            "current_fold": None,
            "total_folds": args.k,
        },
    )
    test_accs = np.array(accuracies["test"], dtype=float)
    train_accs = np.array(accuracies["train"], dtype=float)
    print(
        f"\nSubject-dependent results (per-subject {args.k}-fold):\n"
        f"  Avg Test Acc : {np.mean(test_accs):.2f}%\n"
        f"  Var Test Acc : {np.var(test_accs, ddof=1):.6f}\n"
        f"  Avg Train Acc: {np.mean(train_accs):.2f}%\n"
        f"  Var Train Acc: {np.var(train_accs, ddof=1):.6f}\n"
    )
    plot_subject_dependet(accuracies)


def main():
    args = parse_args()
    # build or reuse run directory
    if args.run_dir:
        run_dir = args.run_dir
        ensure_dir(run_dir)
        # merge config into run dir for traceability
        save_json(os.path.join(run_dir, "config.json"), vars(args))
    else:
        run_dir = build_run_dir(args.results_root, args.model, args.mode, vars(args))
        save_json(os.path.join(run_dir, "config.json"), vars(args))

    if args.mode == "subject_independent":
        run_subject_independent(args, run_dir)
    else:
        run_subject_dependent(args, run_dir)


if __name__ == "__main__":
    main()


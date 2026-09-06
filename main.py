"""
main.py
-------
Master orchestration script to benchmark a model across clinical tasks.

Workflow per task:
1. Load dataset & create patient-isolated splits (Train 70%, Val 15%, Test 15%).
2. Run Optuna HPO on the training split with pruning (8 epochs, 12 trials).
3. Train best configuration for full 30 epochs.
4. Evaluate best checkpoint on held-out test split.
5. Export weights, confusion matrices, performance plots, logs, and README.md.
"""

import os
import gc
import argparse
import torch

from config import PathConfig, ModelConfig
from dataset import build_dataloaders
from optimize import run_hpo
from train import train_model


def run_pipeline(model_name: str, task: str, run_optuna: bool = True):
    """
    Executes the full pipeline for a specific architecture and clinical task.
    """
    print(f"\n{'='*70}")
    print(f"STARTING BENCHMARK: Model = {model_name} | Task = {task}")
    print(f"{'='*70}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Device: {device}")

    # Determine native input image size based on architecture family
    clean_name = model_name.lower().replace("-", "_")
    if "efficientnet" in clean_name:
        img_size = ModelConfig.FAMILY_RESOLUTIONS["efficientnet"]
    elif "densenet" in clean_name:
        img_size = ModelConfig.FAMILY_RESOLUTIONS["densenet"]
    else:
        img_size = ModelConfig.FAMILY_RESOLUTIONS["resnet"]

    print(f"Target Input Resolution: {img_size}x{img_size}")

    # 1. Build DataLoaders & Patient-Isolated Splits
    train_loader, val_loader, test_loader, split_dfs = build_dataloaders(
        task=task,
        img_size=img_size,
        batch_size=ModelConfig.BATCH_SIZE,
        num_workers=ModelConfig.NUM_WORKERS
    )

    print(f"Patient-Isolated Splits Created:")
    print(f"  Train Frames: {len(split_dfs['train'])}")
    print(f"  Val Frames:   {len(split_dfs['val'])}")
    print(f"  Test Frames:  {len(split_dfs['test'])}\n")

    # 2. Hyperparameter Optimization (Optuna)
    if run_optuna:
        best_params = run_hpo(
            model_name=model_name,
            task=task,
            train_loader=train_loader,
            val_loader=val_loader,
            n_trials=ModelConfig.OPTUNA_TRIALS,
            device=device
        )
        lr_head = best_params["lr_head"]
        lr_backbone = best_params["lr_backbone"]
        weight_decay = best_params["weight_decay"]
        label_smoothing = best_params["label_smoothing"]
    else:
        # Default baseline values if Optuna is skipped
        lr_head = 1e-3 if "efficientnet" in clean_name else 3e-4
        lr_backbone = 2e-5 if "efficientnet" in clean_name else 3e-5
        weight_decay = 1e-4
        label_smoothing = 0.05

    # Target folder: /kaggle/working/outputs/{model_name}_{task}
    output_dir = os.path.join(
        PathConfig.OUTPUT_ROOT_DIR,
        f"{model_name}_{task}"
    )

    # 3. Full Model Training (30 Epochs)
    print(f"\n>>> Running Final Full Training ({ModelConfig.NUM_EPOCHS} Epochs) <<<")
    results = train_model(
        model_name=model_name,
        task=task,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=ModelConfig.NUM_EPOCHS,
        lr_head=lr_head,
        lr_backbone=lr_backbone,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        output_dir=output_dir,
        device=device
    )

    print(f"\nFinished Benchmark: {model_name} on {task}")
    print(f"Artifacts and documentation stored in: {output_dir}")

    # Clean up GPU memory
    del train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Cataract Clinical Benchmarking Pipeline")
    parser.add_argument("--model", type=str, default="resnet18",
                        help="Model name (e.g. resnet18, densenet121, efficientnet_b3)")
    parser.add_argument("--task", type=str, default="all", choices=["severity", "poor_dilation", "all"],
                        help="Target dataset/task to run")
    parser.add_argument("--skip-optuna", action="store_true",
                        help="Skip Optuna HPO and run baseline training immediately")

    args = parser.parse_args()

    tasks_to_run = ["severity", "poor_dilation"] if args.task == "all" else [args.task]

    for current_task in tasks_to_run:
        run_pipeline(
            model_name=args.model,
            task=current_task,
            run_optuna=not args.skip_optuna
        )


if __name__ == "__main__":
    main()
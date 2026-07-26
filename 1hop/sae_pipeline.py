#!/usr/bin/env python3
"""
SAE Training and Evaluation Pipeline
Trains SAE models for each transformer layer and evaluates their performance.
"""

import os
import sys
import subprocess
import json
import argparse
from pathlib import Path
from datetime import datetime
import time

class SAEPipeline:
    def __init__(self, base_dir=".", cuda_device=0):
        self.base_dir = Path(base_dir)
        self.cuda_device = cuda_device

        # Configuration
        self.lm_model = self.base_dir / "models/base_sft/final"
        self.output_base = self.base_dir / "models/sae_per_layer"
        self.results_base = self.base_dir / "results/sae_per_layer"

        # Training parameters (paper Appendix C: K=10,000 for 1-hop,
        # Stage 1 = 100 epochs, Stage 2 = 500 epochs)
        self.n_free = 10000
        self.epochs_stage1 = 100
        self.epochs_stage2 = 500
        self.lambda_ortho = "1e-2"

        # Layers to process
        self.layers = list(range(12))  # 0-11

        # Create directories
        self.output_base.mkdir(parents=True, exist_ok=True)
        self.results_base.mkdir(parents=True, exist_ok=True)

    def run_command(self, cmd, description="", env=None):
        """Run a command and stream output live."""
        print(f"Running: {description}")
        print(f"Command: {' '.join(cmd)}")
        print("-" * 50)

        try:
            # Stream output live instead of capturing
            result = subprocess.run(cmd, cwd=self.base_dir, env=env)
            success = result.returncode == 0
            if success:
                print("✓ Success")
            else:
                print("✗ Failed")
                return False
        except Exception as e:
            print(f"✗ Exception: {e}")
            return False

        print("-" * 50)
        return True

    def collect_activations_layer(self, layer):
        """Collect activations for a specific layer (skips if already present)."""
        activation_file = self.base_dir / f"data/activations/train_activations_layer{layer}.pkl"
        if activation_file.exists():
            print(f"✓ Activations already collected: {activation_file}")
            return True

        cmd = [
            "python", "scripts/03_collect_activations.py",
            "--model_path", str(self.lm_model),
            "--layer", str(layer),
            "--output", str(activation_file),
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.cuda_device)

        return self.run_command(cmd, f"Collecting activations for layer {layer}", env=env)

    def train_sae_layer(self, layer):
        """Train SAE for a specific layer."""
        activation_file = self.base_dir / f"data/activations/train_activations_layer{layer}.pkl"
        output_dir = self.output_base / f"layer{layer}"

        if not activation_file.exists():
            print(f"✗ Activation file not found: {activation_file}")
            return False

        output_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing checkpoints to resume from
        resume_checkpoint = None
        final_checkpoint = output_dir / "sae_final.pt"
        if final_checkpoint.exists():
            resume_checkpoint = str(final_checkpoint)
            print(f"Found final checkpoint, will resume training from: {resume_checkpoint}")
        else:
            # Look for the latest stage 2 checkpoint
            stage2_checkpoints = list(output_dir.glob("checkpoint_stage2_epoch_*.pt"))
            if stage2_checkpoints:
                latest_stage2 = max(stage2_checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
                resume_checkpoint = str(latest_stage2)
                print(f"Found latest stage 2 checkpoint, will resume from: {resume_checkpoint}")
            else:
                # Look for latest stage 1 checkpoint
                stage1_checkpoints = list(output_dir.glob("checkpoint_stage1_epoch_*.pt"))
                if stage1_checkpoints:
                    latest_stage1 = max(stage1_checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
                    resume_checkpoint = str(latest_stage1)
                    print(f"Found latest stage 1 checkpoint, will resume from: {resume_checkpoint}")

        cmd = [
            "python", "scripts/04_train_sae.py",
            "--activation_file", str(activation_file),
            "--output_dir", str(output_dir),
            "--mode", "joint",
            "--n_free", str(self.n_free),
            "--epochs_stage1", str(self.epochs_stage1),
            "--epochs_stage2", str(self.epochs_stage2),
            "--lambda_ortho", self.lambda_ortho
        ]

        # Add resume argument if checkpoint exists
        if resume_checkpoint:
            cmd.extend(["--resume", resume_checkpoint])

        # Set CUDA device
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.cuda_device)

        success = self.run_command(cmd, f"Training SAE for layer {layer}", env=env)
        return success

    def evaluate_sae_layer(self, layer):
        """Evaluate SAE for a specific layer."""
        sae_checkpoint = self.output_base / f"layer{layer}/sae_best.pt"
        if not sae_checkpoint.exists():
            # Fall back to final checkpoint if best doesn't exist
            sae_checkpoint = self.output_base / f"layer{layer}/sae_final.pt"
            
        activation_file = self.base_dir / f"data/activations/train_activations_layer{layer}.pkl"
        results_dir = self.results_base / f"layer{layer}"

        if not sae_checkpoint.exists():
            print(f"✗ SAE checkpoint not found: {sae_checkpoint}")
            return False

        results_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python", "scripts/05_evaluate_sae.py",
            "--sae_checkpoint", str(sae_checkpoint),
            "--lm_model", str(self.lm_model),
            "--activation_file", str(activation_file),
            "--layer", str(layer),
            "--output_dir", str(results_dir)
        ]

        success = self.run_command(cmd, f"Evaluating SAE for layer {layer}")
        return success

    def generate_summary(self):
        """Generate a summary report of all results."""
        summary_file = self.results_base / "pipeline_summary.json"

        summary = {
            "pipeline_info": {
                "generated_on": datetime.now().isoformat(),
                "base_dir": str(self.base_dir),
                "lm_model": str(self.lm_model),
                "layers_processed": self.layers,
                "training_params": {
                    "n_free": self.n_free,
                    "epochs_stage1": self.epochs_stage1,
                    "epochs_stage2": self.epochs_stage2,
                    "lambda_ortho": self.lambda_ortho
                }
            },
            "layer_results": {}
        }

        for layer in self.layers:
            results_file = self.results_base / f"layer{layer}/binding_accuracy_results.json"
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        layer_data = json.load(f)

                    summary["layer_results"][str(layer)] = {
                        "train_slot_acc": layer_data.get("train", {}).get("slot_binding_acc", 0),
                        "test_ood_slot_acc": layer_data.get("test_ood", {}).get("slot_binding_acc", 0),
                        "diagonal_acc": layer_data.get("diagonal_accuracy", 0),
                        "swap_success": layer_data.get("swap_controllability", {}).get("best_success_rate", 0),
                        "reconstruction_mse": layer_data.get("reconstruction_mse", 0)
                    }
                except Exception as e:
                    print(f"Error reading results for layer {layer}: {e}")
                    summary["layer_results"][str(layer)] = {"error": str(e)}
            else:
                summary["layer_results"][str(layer)] = {"status": "no_results"}

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Summary saved to: {summary_file}")
        return summary

    def run_pipeline(self, start_layer=None, end_layer=None):
        """Run the complete pipeline."""
        print("=" * 60)
        print("SAE Training and Evaluation Pipeline")
        print("=" * 60)
        print(f"Base directory: {self.base_dir}")
        print(f"LM model: {self.lm_model}")
        print(f"CUDA device: {self.cuda_device}")
        print()

        # Filter layers if specified
        layers_to_process = self.layers
        if start_layer is not None:
            layers_to_process = [l for l in layers_to_process if l >= start_layer]
        if end_layer is not None:
            layers_to_process = [l for l in layers_to_process if l <= end_layer]

        print(f"Processing layers: {layers_to_process}")
        print()

        results = {}

        for layer in layers_to_process:
            print(f"Processing layer {layer}...")
            start_time = time.time()

            # Collect activations (if needed), then train SAE
            collect_success = self.collect_activations_layer(layer)
            train_success = self.train_sae_layer(layer) if collect_success else False

            # Evaluate SAE (only if training succeeded)
            eval_success = False
            if train_success:
                eval_success = self.evaluate_sae_layer(layer)

            elapsed = time.time() - start_time
            results[layer] = {
                "train_success": train_success,
                "eval_success": eval_success,
                "elapsed_time": elapsed
            }

            print(f"Layer {layer} completed in {elapsed:.1f} seconds")
            print("-" * 50)
        # Generate summary
        print("\nGenerating summary report...")
        summary = self.generate_summary()

        # Print final summary
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED")
        print("=" * 60)

        successful_layers = [l for l, r in results.items() if r["train_success"] and r["eval_success"]]
        print(f"Successfully processed layers: {successful_layers}")

        if summary["layer_results"]:
            print("\nPerformance Summary:")
            print("Layer | Train Slot Acc | Test Slot Acc | Diagonal Acc | Swap Success")
            print("------|---------------|---------------|--------------|-------------")
            for layer_str, data in summary["layer_results"].items():
                if "train_slot_acc" in data:
                    print(f"{int(layer_str):2d}    | {data['train_slot_acc']:6.3f}         | {data['test_ood_slot_acc']:6.3f}         | {data['diagonal_acc']:6.3f}        | {data['swap_success']:6.3f}")

        print(f"\nDetailed results saved to: {self.results_base}")
        print("=" * 60)

        return results

def main():
    parser = argparse.ArgumentParser(description="SAE Training and Evaluation Pipeline")
    parser.add_argument("--base_dir", type=str, default=".",
                       help="Base directory for the project (the 1hop/ directory)")
    parser.add_argument("--cuda_device", type=int, default=0,
                       help="CUDA device to use")
    parser.add_argument("--start_layer", type=int, default=None,
                       help="Start layer (inclusive)")
    parser.add_argument("--end_layer", type=int, default=None,
                       help="End layer (inclusive)")
    parser.add_argument("--test_layer", type=int, default=None,
                       help="Test only this layer")
    parser.add_argument("--evaluate_all", action="store_true",
                       help="Evaluate all layers (0-11) in parallel")

    args = parser.parse_args()

    pipeline = SAEPipeline(args.base_dir, args.cuda_device)
    
    if args.evaluate_all:
        # Run evaluation on all layers
        print("Running evaluation on all layers...")
        results = {}
        for layer in range(12):  # 0-11
            print(f"Evaluating layer {layer}...")
            success = pipeline.evaluate_sae_layer(layer)
            results[layer] = {"eval_success": success}
            if success:
                print(f"✓ Layer {layer} evaluation complete")
            else:
                print(f"✗ Layer {layer} evaluation failed")
        
        # Generate summary
        print("\nGenerating final summary...")
        summary = pipeline.generate_summary()
        
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print(f"Results saved to: {pipeline.results_base}")
        
    else:
        # If test_layer is specified, only process that layer
        if args.test_layer is not None:
            args.start_layer = args.end_layer = args.test_layer

        results = pipeline.run_pipeline(args.start_layer, args.end_layer)

if __name__ == "__main__":
    main()
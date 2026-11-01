import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any

import config

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def load_evaluation_data() -> List[Dict[str, Any]]:
    eval_path = config.METRICS_DIR / "all_models_evaluation.json"
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation metrics not found at: {eval_path}. Run train_models.py first.")
    with open(eval_path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_comparison_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    records = []
    for r in results:
        records.append({
            "Model Name": r["model_name"],
            "Architecture Type": r["type"],
            "Val Accuracy": round(r["val_accuracy"], 4),
            "Test Accuracy": round(r["test_accuracy"], 4),
            "Macro Precision": round(r["precision_macro"], 4),
            "Macro Recall": round(r["recall_macro"], 4),
            "Macro F1-Score": round(r["f1_macro"], 4),
            "Train Time (s)": round(r["train_time_sec"], 2),
            "Inference Time (ms)": round(r["inference_time_ms"], 2),
            "Epochs Trained": r.get("epochs_trained", 0)
        })
    df = pd.DataFrame(records)
    
    # Save CSV and JSON
    df.to_csv(config.METRICS_DIR / "model_comparison.csv", index=False)
    df.to_json(config.METRICS_DIR / "model_comparison.json", orient="records", indent=2)
    return df

def plot_training_histories():
    dl_models = ["simplernn", "lstm", "gru", "bilstm"]
    histories = {}
    
    for m in dl_models:
        path = config.METRICS_DIR / f"{m}_history.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                histories[m.upper()] = json.load(f)
                
    if not histories:
        print("No DL histories found to plot.")
        return

    # 1. Training & Validation Accuracy Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Train & Val Accuracy
    for name, hist in histories.items():
        epochs = range(1, len(hist["accuracy"]) + 1)
        axes[0].plot(epochs, hist["accuracy"], label=f"{name} (Train)", linestyle="--", alpha=0.7)
        axes[0].plot(epochs, hist["val_accuracy"], label=f"{name} (Val)", linewidth=2)
    axes[0].set_title("Model Accuracy Across Epochs", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Train & Val Loss
    for name, hist in histories.items():
        epochs = range(1, len(hist["loss"]) + 1)
        axes[1].plot(epochs, hist["loss"], label=f"{name} (Train)", linestyle="--", alpha=0.7)
        axes[1].plot(epochs, hist["val_loss"], label=f"{name} (Val)", linewidth=2)
    axes[1].set_title("Model Loss Across Epochs", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Sparse Categorical Crossentropy Loss")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plot_path = config.GRAPHS_DIR / "dl_learning_curves.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved learning curves plot to: {plot_path}")

def plot_metrics_comparison(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(df))
    width = 0.2

    ax.bar(x - 1.5 * width, df["Test Accuracy"], width, label="Test Accuracy", color="#2563eb")
    ax.bar(x - 0.5 * width, df["Macro Precision"], width, label="Macro Precision", color="#059669")
    ax.bar(x + 0.5 * width, df["Macro Recall"], width, label="Macro Recall", color="#d97706")
    ax.bar(x + 1.5 * width, df["Macro F1-Score"], width, label="Macro F1-Score", color="#7c3aed")

    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11, fontweight="bold")
    ax.set_title("Comprehensive Performance Comparison Across All Models", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model Name"], rotation=15, ha="right", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.7)

    plt.tight_layout()
    plot_path = config.GRAPHS_DIR / "metrics_comparison_bar.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved metrics comparison bar plot to: {plot_path}")

def plot_time_latency_comparison(df: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    colors = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"]

    # Training time
    ax1.set_xticks(range(len(df)))
    bars1 = ax1.bar(df["Model Name"], df["Train Time (s)"], color=colors[:len(df)])
    ax1.set_title("Total Training Time (Seconds)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Seconds")
    ax1.set_xticklabels(df["Model Name"], rotation=20, ha="right", fontsize=8)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{yval:.1f}s", ha="center", va="bottom", fontsize=8)

    # Inference latency
    ax2.set_xticks(range(len(df)))
    bars2 = ax2.bar(df["Model Name"], df["Inference Time (ms)"], color=colors[:len(df)])
    ax2.set_title("Inference Latency per Query (Milliseconds)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Milliseconds (ms)")
    ax2.set_xticklabels(df["Model Name"], rotation=20, ha="right", fontsize=8)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{yval:.1f}ms", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plot_path = config.GRAPHS_DIR / "time_and_latency_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved latency/time comparison plot to: {plot_path}")

def plot_confusion_matrices(results: List[Dict[str, Any]]):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, r in enumerate(results):
        cm = np.array(r["confusion_matrix"])
        ax = axes[idx]
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f"{r['model_name']}\nTest Acc: {r['test_accuracy']:.2f} | F1: {r['f1_macro']:.2f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted Label Index", fontsize=8)
        ax.set_ylabel("True Label Index", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide extra unused subplot if 5 models
    if len(results) < len(axes):
        for i in range(len(results), len(axes)):
            fig.delaxes(axes[i])

    plt.tight_layout()
    plot_path = config.GRAPHS_DIR / "confusion_matrices_all.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved all confusion matrices plot to: {plot_path}")

def run_evaluation():
    print("=" * 60)
    print("SRKR CHATBOT FAIR MODEL EVALUATION & ANALYSIS")
    print("=" * 60)

    results = load_evaluation_data()
    df = generate_comparison_table(results)

    print("\nMODEL PERFORMANCE COMPARISON TABLE:")
    print("-" * 60)
    print(df.to_string(index=False))
    print("-" * 60)

    # Model selection based on validation accuracy and F1 score
    best_model_row = df.sort_values(by=["Val Accuracy", "Macro F1-Score"], ascending=False).iloc[0]
    print(f"\n[BEST PERFORMING MODEL]: {best_model_row['Model Name']}")
    print(f"  Validation Accuracy: {best_model_row['Val Accuracy']}")
    print(f"  Test Accuracy:       {best_model_row['Test Accuracy']}")
    print(f"  Macro F1-Score:      {best_model_row['Macro F1-Score']}")
    print(f"  Inference Latency:   {best_model_row['Inference Time (ms)']} ms")

    # Generate graphs
    plot_training_histories()
    plot_metrics_comparison(df)
    plot_time_latency_comparison(df)
    plot_confusion_matrices(results)

    print("\nAll evaluation artifacts and comparison charts generated successfully!")

if __name__ == "__main__":
    run_evaluation()

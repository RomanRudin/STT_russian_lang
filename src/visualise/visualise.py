import json
import os
import matplotlib.pyplot as plt

def plot_and_save_metrics(json_path, output_dir=None):
    with open(json_path, 'r') as f:
        data = json.load(f)

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(json_path) or ".", "visualise")
    os.makedirs(output_dir, exist_ok=True)

    log = data["log_history"]
    best_metric = data.get("best_metric")
    best_checkpoint = data.get("best_model_checkpoint")

    epoch_train_loss = {}
    epoch_eval = {}

    for entry in log:
        epoch = entry.get("epoch")
        if epoch is None:
            continue
        epoch_int = int(epoch)
        if "loss" in entry and "eval_loss" not in entry:
            epoch_train_loss[epoch_int] = entry["loss"]
        elif "eval_loss" in entry:
            epoch_eval[epoch_int] = {
                "eval_loss": entry["eval_loss"],
                "eval_wer": entry.get("eval_wer"),
                "eval_cer": entry.get("eval_cer"),
                "eval_f1": entry.get("eval_f1"),
                "eval_ser": entry.get("eval_ser"),
                "eval_space_wer": entry.get("eval_space_wer"),
                "eval_hits": entry.get("eval_hits"),
                "eval_substitutions": entry.get("eval_substitutions"),
                "eval_deletions": entry.get("eval_deletions"),
                "eval_insertions": entry.get("eval_insertions"),
                "eval_runtime": entry.get("eval_runtime"),
            }

    sorted_epochs = sorted(set(epoch_train_loss.keys()) | set(epoch_eval.keys()))
    epochs = sorted_epochs

    train_loss = [epoch_train_loss.get(e) for e in epochs]
    eval_loss = [epoch_eval[e]["eval_loss"] if e in epoch_eval else None for e in epochs]
    wer = [epoch_eval[e]["eval_wer"] if e in epoch_eval else None for e in epochs]
    cer = [epoch_eval[e]["eval_cer"] if e in epoch_eval else None for e in epochs]
    f1 = [epoch_eval[e]["eval_f1"] if e in epoch_eval else None for e in epochs]
    ser = [epoch_eval[e]["eval_ser"] if e in epoch_eval else None for e in epochs]
    space_wer = [epoch_eval[e]["eval_space_wer"] if e in epoch_eval else None for e in epochs]
    hits = [epoch_eval[e]["eval_hits"] if e in epoch_eval else None for e in epochs]
    subs = [epoch_eval[e]["eval_substitutions"] if e in epoch_eval else None for e in epochs]
    dels = [epoch_eval[e]["eval_deletions"] if e in epoch_eval else None for e in epochs]
    inss = [epoch_eval[e]["eval_insertions"] if e in epoch_eval else None for e in epochs]
    eval_runtime = [epoch_eval[e]["eval_runtime"] if e in epoch_eval else None for e in epochs]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, 'o-', label="Train Loss", color="tab:blue")
    ax.plot(epochs, eval_loss, 's-', label="Eval Loss", color="tab:orange")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    ax.set_xticks(epochs)
    ax.set_title("Loss")

    # 2. WER / CER
    ax = axes[0, 1]
    ax.plot(epochs, wer, 'o-', label="WER", color="tab:red")
    ax.plot(epochs, cer, 's-', label="CER", color="tab:green")
    ax.set_ylabel("Error Rate")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)
    ax.set_xticks(epochs)
    ax.set_title("Word & Character Error Rate")

    # 3. F1 / SER / Space WER
    ax = axes[1, 0]
    ax.plot(epochs, f1, '^-', label="F1", color="tab:purple")
    ax.plot(epochs, ser, 'v-', label="SER", color="tab:brown")
    ax.plot(epochs, space_wer, 'd-', label="Space WER", color="tab:cyan")
    ax.set_ylabel("Metric Value")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)
    ax.set_xticks(epochs)
    ax.set_title("F1, Sentence Error Rate, Space WER")

    # 4. Edit counts
    ax = axes[1, 1]
    ax.plot(epochs, hits, 'o-', label="Hits", color="tab:green")
    ax.plot(epochs, subs, 's-', label="Substitutions", color="tab:red")
    ax.plot(epochs, dels, '^-', label="Deletions", color="tab:orange")
    ax.plot(epochs, inss, 'v-', label="Insertions", color="tab:blue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True)
    ax.set_xticks(epochs)
    ax.set_title("Edit Operations")

    plt.tight_layout()
    graph_path = os.path.join(output_dir, "metrics_by_epoch.png")
    plt.savefig(graph_path, dpi=150)
    plt.close()
    print(f"Graph saved to {graph_path}")

    # Table
    table_path = os.path.join(output_dir, "metrics_table.txt")
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write("Metrics per epoch\n")
        f.write("=" * 100 + "\n")
        if best_metric is not None:
            f.write(f"Best metric: {best_metric}\n")
        if best_checkpoint is not None:
            f.write(f"Best checkpoint: {best_checkpoint}\n")
        f.write("-" * 100 + "\n")
        header = f"{'Epoch':<6} {'Train Loss':<12} {'Eval Loss':<12} {'WER':<8} {'CER':<8} {'F1':<8} {'SER':<8} {'Space WER':<10} {'Hits':<8} {'Subs':<8} {'Dels':<8} {'Inss':<8} {'Eval Runtime (s)':<16}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for i, e in enumerate(epochs):
            tl = f"{train_loss[i]:.4f}" if train_loss[i] is not None else "N/A"
            el = f"{eval_loss[i]:.4f}" if eval_loss[i] is not None else "N/A"
            w = f"{wer[i]:.4f}" if wer[i] is not None else "N/A"
            c = f"{cer[i]:.4f}" if cer[i] is not None else "N/A"
            f1s = f"{f1[i]:.4f}" if f1[i] is not None else "N/A"
            se = f"{ser[i]:.4f}" if ser[i] is not None else "N/A"
            sw = f"{space_wer[i]:.4f}" if space_wer[i] is not None else "N/A"
            hi = f"{hits[i]:.0f}" if hits[i] is not None else "N/A"
            su = f"{subs[i]:.0f}" if subs[i] is not None else "N/A"
            de = f"{dels[i]:.0f}" if dels[i] is not None else "N/A"
            ins = f"{inss[i]:.0f}" if inss[i] is not None else "N/A"
            rt = f"{eval_runtime[i]:.2f}" if eval_runtime[i] is not None else "N/A"
            row = f"{e:<6} {tl:<12} {el:<12} {w:<8} {c:<8} {f1s:<8} {se:<8} {sw:<10} {hi:<8} {su:<8} {de:<8} {ins:<8} {rt:<16}"
            f.write(row + "\n")
        f.write("-" * len(header) + "\n")
        total_eval_time = sum(x for x in eval_runtime if x is not None)
        f.write(f"Total evaluation time: {total_eval_time:.2f} s\n")
    print(f"Metrics table saved to {table_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python plot_metrics.py <trainer_state.json> [output_dir]")
    else:
        out = sys.argv[2] if len(sys.argv) > 2 else None
        plot_and_save_metrics(sys.argv[1], out)
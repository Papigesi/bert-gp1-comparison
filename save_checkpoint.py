import os

def save_checkpoint(model, tokenizer, epoch, val_metrics, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}")
    os.makedirs(checkpoint_path, exist_ok=True)

    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)

    with open(os.path.join(checkpoint_path, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"epoch: {epoch}\n")
        for key, value in val_metrics.items():
            f.write(f"{key}: {value}\n")

    return checkpoint_path
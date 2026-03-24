import os
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from data_loader import load_imdb_data
from tokenization import BERTIMDBDataset, bert_tokenizer
from evaluate_model import evaluate_model
from save_checkpoint import save_checkpoint


MODEL_NAME = "bert-base-uncased"

MAX_LENGTH = 256
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

CHECKPOINT_DIR = "./checkpoints_bert"
FINAL_MODEL_DIR = "./final_bert_imdb"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()

    train_df, val_df, test_df = load_imdb_data()

    train_dataset = BERTIMDBDataset(
        texts=train_df["text"],
        labels=train_df["label"],
        tokenizer=bert_tokenizer,
        max_length=MAX_LENGTH
    )

    val_dataset = BERTIMDBDataset(
        texts=val_df["text"],
        labels=val_df["label"],
        tokenizer=bert_tokenizer,
        max_length=MAX_LENGTH
    )

    test_dataset = BERTIMDBDataset(
        texts=test_df["text"],
        labels=test_df["label"],
        tokenizer=bert_tokenizer,
        max_length=MAX_LENGTH
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False
    )

    print("Dataset is ready.")
    print("The classification BERT model is being preparing...")
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    total_training_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_training_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_f1 = -1.0
    best_checkpoint_path = None

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"{'=' * 60}")

        model.train()
        total_train_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=True)

        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                
            total_train_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if step % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{current_lr:.2e}"
                })

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"\nAverage training loss: {avg_train_loss:.4f}")

        val_metrics = evaluate_model(model, val_loader, device, use_amp)
        print("Validation results:")
        print(f"  Loss      : {val_metrics['loss']:.4f}")
        print(f"  Accuracy  : {val_metrics['accuracy']:.4f}")
        print(f"  Precision : {val_metrics['precision']:.4f}")
        print(f"  Recall    : {val_metrics['recall']:.4f}")
        print(f"  F1 Score  : {val_metrics['f1']:.4f}")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_checkpoint_path = save_checkpoint(
                model=model,
                tokenizer=bert_tokenizer,
                epoch=epoch,
                val_metrics=val_metrics,
                checkpoint_dir=CHECKPOINT_DIR
            )
            print(f"Best model updated and saved to: {best_checkpoint_path}")

    print(f"\nBest validation F1: {best_val_f1:.4f}")
    if best_checkpoint_path is not None:
        print(f"Loading best checkpoint from: {best_checkpoint_path}")
        model = BertForSequenceClassification.from_pretrained(best_checkpoint_path)
        model.to(device)

    test_metrics = evaluate_model(model, test_loader, device, use_amp)
    print("\nFinal test results:")
    print(f"  Loss      : {test_metrics['loss']:.4f}")
    print(f"  Accuracy  : {test_metrics['accuracy']:.4f}")
    print(f"  Precision : {test_metrics['precision']:.4f}")
    print(f"  Recall    : {test_metrics['recall']:.4f}")
    print(f"  F1 Score  : {test_metrics['f1']:.4f}")

    print(f"\nSaving final model to: {FINAL_MODEL_DIR}")
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    model.save_pretrained(FINAL_MODEL_DIR)
    bert_tokenizer.save_pretrained(FINAL_MODEL_DIR)

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    print("Torch Version   :", torch.__version__)
    print("CUDA Available  :", torch.cuda.is_available())
    print("CUDA Version    :", torch.version.cuda)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("GPU: Not Found")

    main()
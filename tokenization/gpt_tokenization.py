import torch
from torch.utils.data import Dataset
from transformers import OpenAIGPTTokenizer


gpt_tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")

if gpt_tokenizer.pad_token is None:
    gpt_tokenizer.add_special_tokens({"pad_token": "[PAD]"})


class GPTIMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256, use_prompt=True):
        self.texts = list(texts)
        self.labels = list(labels)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_prompt = use_prompt

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]

        if self.use_prompt:
            text = f"Review: {text} Sentiment:"

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {key: value.squeeze(0) for key, value in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item
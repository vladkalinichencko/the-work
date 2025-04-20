import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from datasets import load_dataset
import wandb
import time
import matplotlib.pyplot as plt

# --- Настройки ---
BATCH_SIZE = 2
MAX_LEN = 64
LOG_INTERVAL = 100  # логировать каждые 100 шагов
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 1

# --- W&B ---
wandb.init(project="gpt2_causal_vs_regular", name="gpt2_comparison", reinit=True)

# --- Датасет ---
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train[:1%]')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def encode(example):
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=MAX_LEN)

dataset = dataset.map(encode)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Модели ---
config = GPT2Config()
regular_gpt2 = GPT2LMHeadModel(config).to(DEVICE)
# --- Каузальные эмбеддинги ---
from implementation.notebooks.causal_embeddings import CausalEmbeddingModel
causal_embeds = torch.load('../models/causal_embeds.pth')
causal_gpt2 = GPT2LMHeadModel(config).to(DEVICE)
with torch.no_grad():
    causal_gpt2.transformer.wte.weight.copy_(causal_embeds['embeddings'])

# --- Оптимизаторы ---
optim_regular = torch.optim.AdamW(regular_gpt2.parameters(), lr=5e-5)
optim_causal = torch.optim.AdamW(causal_gpt2.parameters(), lr=5e-5)

# --- Тренировка ---
def train(model, optimizer, tag):
    model.train()
    losses = []
    step = 0
    start_time = time.time()
    for epoch in range(EPOCHS):
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if step % LOG_INTERVAL == 0:
                wandb.log({f"loss_{tag}": loss.item(), "step": step, "time": time.time() - start_time})
            step += 1
    return losses

losses_regular = train(regular_gpt2, optim_regular, tag="regular")
losses_causal = train(causal_gpt2, optim_causal, tag="causal")

# --- Визуализация ---
plt.plot(losses_regular, label='Regular GPT-2')
plt.plot(losses_causal, label='Causal Embeddings GPT-2')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss Dynamics')
plt.show()

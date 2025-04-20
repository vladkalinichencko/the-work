#!/usr/bin/env python3
import os
import time
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup, DataCollatorForLanguageModeling
from datasets import load_dataset
import matplotlib.pyplot as plt

# импорт CausalSelfAttention из notebook
from implementation.notebooks.attention_integration import CausalSelfAttention

# Определение GPT2 с causal-attention + LM head
default_config = GPT2Config()
class CausalGPT2LMHeadModel(GPT2LMHeadModel):
  def __init__(self, config):
    super().__init__(config)
    for block in self.transformer.h:
      block.attn = CausalSelfAttention(config)

# Функция обучения и оценки
def train_model(model, tokenizer, train_dataset, val_dataset, epochs=3, batch_size=8, lr=5e-5, freeze_alpha=False):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  # freeze alpha parameter if requested
  if freeze_alpha:
    for name, p in model.named_parameters():
      if 'alpha' in name:
        p.requires_grad = False
  data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
  val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, collate_fn=data_collator)
  optimizer = AdamW(model.parameters(), lr=lr)
  total_steps = len(train_loader) * epochs
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
  losses = []
  start = time.time()
  for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
      inputs = batch['input_ids'].to(device)
      labels = inputs.clone()
      outputs = model(inputs, labels=labels)
      loss = outputs.loss
      loss.backward()
      optimizer.step()
      scheduler.step()
      optimizer.zero_grad()
      epoch_loss += loss.item()
    avg = epoch_loss / len(train_loader)
    losses.append(avg)
    print(f'Epoch {epoch+1}/{epochs} | loss {avg:.4f} | time {time.time()-start:.1f}s')
  # оценка perplexity
  model.eval()
  eval_loss = 0
  with torch.no_grad():
    for batch in val_loader:
      inputs = batch['input_ids'].to(device)
      labels = inputs.clone()
      outputs = model(inputs, labels=labels)
      eval_loss += outputs.loss.item()
  eval_loss /= len(val_loader)
  ppl = torch.exp(torch.tensor(eval_loss))
  print(f'Validation loss {eval_loss:.4f} | perplexity {ppl:.1f}')
  return losses, eval_loss, ppl

# Основная функция
if __name__ == '__main__':
  # подготовка папки для результатов
  eval_dir = os.path.join(os.path.dirname(__file__), '..', 'evaluation')
  os.makedirs(eval_dir, exist_ok=True)

  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  # загрузка датасета
  raw = load_dataset('wikitext', 'wikitext-2-raw-v1')
  def tok(examples): return tokenizer(examples['text'])
  train_ds = raw['train'].map(tok, batched=True, remove_columns=['text'])
  val_ds = raw['validation'].map(tok, batched=True, remove_columns=['text'])

  # baseline model with GPT2 pretrained embeddings
  print('Setting up baseline GPT2 with pretrained embeddings...')
  pretrained_full = GPT2LMHeadModel.from_pretrained('gpt2')
  orig_model = GPT2LMHeadModel(default_config)
  orig_model.transformer.wte.weight.data.copy_(pretrained_full.transformer.wte.weight.data)
  print('Training baseline GPT2...')
  orig_losses, orig_eval_loss, orig_ppl = train_model(orig_model, tokenizer, train_ds, val_ds)

  # causal model with trained causal embeddings
  print('Setting up causal GPT2 with causal embeddings...')
  causal_model = CausalGPT2LMHeadModel(default_config)
  embed_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'causal_embeds.pth')
  causal_embeds = torch.load(embed_path, map_location='cpu')
  causal_model.transformer.wte.weight.data.copy_(causal_embeds)
  print('Training causal GPT2...')
  causal_losses, causal_eval_loss, causal_ppl = train_model(causal_model, tokenizer, train_ds, val_ds, freeze_alpha=True)

  # визуализация кривых обучения
  plt.figure()
  plt.plot(range(1, len(orig_losses)+1), orig_losses, label='Original')
  plt.plot(range(1, len(causal_losses)+1), causal_losses, label='Causal')
  plt.xlabel('Epoch')
  plt.ylabel('Training loss')
  plt.legend()
  plt.savefig(os.path.join(eval_dir, 'train_loss_comparison.png'))

  # сохранение метрик
  with open(os.path.join(eval_dir, 'metrics.txt'), 'w') as f:
    f.write(f'Original | val_loss: {orig_eval_loss:.4f} | ppl: {orig_ppl:.1f}\n')
    f.write(f'Causal  | val_loss: {causal_eval_loss:.4f} | ppl: {causal_ppl:.1f}\n')

  print('Comparison complete. Результаты в', eval_dir)

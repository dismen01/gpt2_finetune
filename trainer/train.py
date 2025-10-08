import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from torch.utils.data import Dataset
import glob
import os

# ==== Настройка устройства и директорий ====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Используем GPU, если есть
DATA_DIR = "data"  # Папка с текстовыми файлами для обучения
MODEL_DIR = "rugpt3_finetuned"  # Папка, куда сохраняем финетюн модель

# ==== 1) Загружаем базовую модель и токенизатор ====
# Используем предобученный RuGPT3 large от Sberbank
tokenizer_base = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")
model_base = AutoModelForCausalLM.from_pretrained(
    "sberbank-ai/rugpt3large_based_on_gpt2"
).to(DEVICE)  # Переносим модель на GPU/CPU

# ==== 2) Dataset ====
# Класс для подготовки данных к обучению модели
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, files, tokenizer, block_size=128):
        self.examples = []
        self.block_size = block_size

        for f in files:
            # Читаем текст из файла
            with open(f, "r", encoding="utf-8") as fh:
                text = fh.read().strip()

            # Токенизируем текст базовой моделью
            encodings = tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=block_size * 200,  # ограничение по длине
            )
            input_ids = encodings["input_ids"]

            # Нарезаем на блоки фиксированной длины (block_size)
            for i in range(0, len(input_ids), block_size):
                block = input_ids[i : i + block_size]
                if len(block) == block_size:
                    self.examples.append(torch.tensor(block, dtype=torch.long))

    def __len__(self):
        # Общее количество блоков
        return len(self.examples)

    def __getitem__(self, idx):
        # Возвращаем блок как словарь: input_ids и labels (для language modeling)
        return {"input_ids": self.examples[idx], "labels": self.examples[idx]}

# ==== Загружаем все текстовые файлы из папки data ====
text_files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
dataset = TextDataset(text_files, tokenizer_base, block_size=128)

# ==== 3) Data collator ====
# Отвечает за формирование батчей для обучения, mlm=False для causal LM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer_base, mlm=False)

# ==== 4) TrainingArguments и Trainer ====
# Параметры обучения
training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    overwrite_output_dir=True,
    num_train_epochs=5,  # количество эпох
    per_device_train_batch_size=4,  # размер батча для GPU
    save_steps=500,  # сохраняем модель каждые 500 шагов
    save_total_limit=2,  # сохраняем только 2 последние версии модели
    prediction_loss_only=True,
    logging_steps=50,
    fp16=torch.cuda.is_available(),  # смешанная точность, если GPU
)

# Trainer — обёртка для обучения модели
trainer = Trainer(
    model=model_base,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# ==== 5) Fine-tune (опционально) ====
trainer.train()  # запуск обучения
trainer.save_model(MODEL_DIR)  # сохраняем модель
tokenizer_base.save_pretrained(MODEL_DIR)  # сохраняем токенизатор

# ==== 6) Генерация текста ====
from transformers import pipeline

# Создаём pipeline для генерации текста
pipe = pipeline(
    "text-generation",
    model=model_base,
    tokenizer=tokenizer_base,
    device=0 if DEVICE == "cuda" else -1,  # устройство: GPU/CPU
)

# Пример генерации
prompt = "Юнг писал, что человеческое сознание часто проявляется через символы."
gen = pipe(
    prompt, max_length=100, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1
)
print("Generated:", gen[0]["generated_text"])  # выводим сгенерированный текст

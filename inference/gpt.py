import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel

# ==== Пути к моделям ====
trained_model_dir = "./rugpt3_finetuned"  # Папка с твоей дообученной моделью

# ==== 1) Загрузка русскоязычной модели (голая RuGPT3) ====
# Предобученная модель от Sberbank, используем как базу
tokenizer_base = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")
model_base = AutoModelForCausalLM.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")
model_base.eval()  # переводим в режим оценки (без обучения)

# ==== 2) Загрузка дообученной модели ====
# Попытка загрузить токенизатор из папки с финетюном
try:
    tokenizer_trained = AutoTokenizer.from_pretrained(trained_model_dir)
except:
    # Если токенизатор не найден, используем стандартный GPT-2
    tokenizer_trained = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer_trained.pad_token = tokenizer_trained.eos_token  # добавляем токен паддинга

# Загрузка модели, которую мы дообучали
model_trained = GPT2LMHeadModel.from_pretrained(trained_model_dir)
model_trained.eval()  # режим оценки

# ==== 3) Функция генерации текста ====
def generate_text(model, tokenizer, prompt, max_length=100):
    # Токенизация входного текста
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=100)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():  # отключаем градиенты (только инференс)
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            do_sample=True,  # включаем случайность в генерации
            temperature=0.8,  # регулируем креативность
            top_k=50,         # фильтр на топ-k вероятностей
            top_p=0.9,        # фильтр на топ-p (nucleus sampling)
            repetition_penalty=1.2,  # штраф за повторения
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=True  # остановка при достижении конца последовательности
        )
    # Декодируем сгенерированные токены в текст
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ==== 4) Пример интерактивного чата ====
print("\n--- Чат: сначала голая русскоязычная GPT-2, потом дообученная ---")
print("Для выхода напиши 'exit' или 'выход'\n")

while True:
    prompt = input("Ты: ")
    if prompt.lower() in ["exit", "выход"]:
        break

    # --- Голая модель ---
    base_text = generate_text(model_base, tokenizer_base, prompt)
    print(f"\033[94mГолая GPT-2:\033[0m {base_text}")

    # --- Дообученная модель ---
    trained_text = generate_text(model_trained, tokenizer_trained, prompt)
    print(f"\033[92mДообученная модель:\033[0m {trained_text}\n")

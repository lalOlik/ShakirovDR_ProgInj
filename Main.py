"""
Умный Фото-Анализатор + Генератор Описаний
Приложение для Github

Приложение объединяет две нейросети (можно загрузить своё фото и использовать):
1. CLIP - для анализа изображений
2. BLIP - для генерации описаний
"""

# ============================================================================ 
# УСТАНОВКА БИБЛИОТЕК
# ============================================================================

print("Установка зависимостей...")

!pip install -q ftfy regex tqdm
!pip install -q git+https://github.com/openai/CLIP.git
!pip install -q transformers torch torchvision pillow requests accelerate
!pip install -q deep-translator

print("Все библиотеки установлены!\n")

# ============================================================================ 
# ИМПОРТ БИБЛИОТЕК
# ============================================================================

import torch
import clip
from PIL import Image
import requests
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
from IPython.display import display
from deep_translator import GoogleTranslator
from google.colab import files
import warnings
warnings.filterwarnings('ignore')

print("Библиотеки импортированы\n")

translator = GoogleTranslator(source='en', target='ru')

# ============================================================================ 
# ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ
# ============================================================================

print("ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ...\n")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используется устройство: {device}")

# ---- CLIP ----
print("\nЗагрузка CLIP...")
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()
print("CLIP загружен")

# ---- BLIP ----
print("\nЗагрузка BLIP...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)
blip_model.eval()
print("BLIP загружен")

print("\nВсе модели готовы!\n")

# ============================================================================ 
# ФУНКЦИИ
# ============================================================================

def load_image(image_path_or_url):
    """Загрузка изображения"""
    try:
        if isinstance(image_path_or_url, str) and image_path_or_url.startswith('http'):
            r = requests.get(image_path_or_url, timeout=10)
            img = Image.open(BytesIO(r.content)).convert('RGB')
        else:
            img = Image.open(image_path_or_url).convert('RGB')
        return img
    except Exception as e:
        print(f"Ошибка загрузки изображения: {e}")
        return None

def analyze_image_with_clip(image, categories):
    """Анализ изображения CLIP-ом"""
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of {c}") for c in categories]).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    return similarity[0].cpu().numpy()

def generate_caption_blip(image):
    """Генерация точного описания BLIP"""
    inputs = blip_processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = blip_model.generate(**inputs, max_length=70)

    caption_en = blip_processor.decode(output[0], skip_special_tokens=True)

    try:
        caption_ru = translator.translate(caption_en)
        return caption_ru, caption_en
    except:
        return caption_en, caption_en

def smart_photo_analyzer(image_path_or_url, custom_categories=None, show_top_n=5):
    print("\n" + "="*70)
    print("АНАЛИЗ ИЗОБРАЖЕНИЯ")
    print("="*70)

    # ---- Загрузка изображения ----
    image = load_image(image_path_or_url)
    if image is None:
        print("Не удалось загрузить изображение.")
        return

    display(image.resize((400, 400)))

    # ---- Категории ----
    if custom_categories is None:
        categories = [
            "a cat", "a dog", "a person", "a car", "a nature landscape",
            "a city skyline", "food", "an interior design", "an animal",
            "technology device", "a sunset", "a beach", "mountains",
            "a forest", "a building", "art", "flowers"
        ]
    else:
        categories = custom_categories

    print("\nАнализ с помощью CLIP...")
    probabilities = analyze_image_with_clip(image, categories)
    sorted_idx = np.argsort(probabilities)[::-1]

    # ---- Вывод TOP ----
    print(f"\nТОП-{show_top_n} результатов:")
    print("-" * 70)

    results = []
    for i in range(min(show_top_n, len(categories))):
        idx = sorted_idx[i]
        prob = probabilities[idx]
        cat = categories[idx]

        bar = "█" * int(prob * 50)
        bar += "░" * (50 - len(bar))

        print(f"{i+1}. {cat:25s} {bar} {prob*100:5.2f}%")
        results.append((cat, float(prob)))

    top_category = categories[sorted_idx[0]]
    top_confidence = probabilities[sorted_idx[0]] * 100

    print("\n" + "="*70)
    print(f"Лучшая категория: {top_category}")
    print(f"Уверенность: {top_confidence:.2f}%")
    print("="*70)

    # ---- Описание BLIP ----
    print("\nГенерация описания BLIP...")
    description_ru, description_en = generate_caption_blip(image)

    print("\nСгенерированное описание:")
    print("-" * 70)
    print("EN:", description_en)
    print("RU:", description_ru)
    print("-" * 70)

    return {
        "top_category": top_category,
        "confidence": float(top_confidence),
        "clip_results": results,
        "caption_ru": description_ru,
        "caption_en": description_en,
        "image": image
    }

# ============================================================================ 
# ЗАГРУЗКА СВОЕГО ИЗОБРАЖЕНИЯ
# ============================================================================

print("Загрузите свое изображение:")

uploaded = files.upload()
for file_name in uploaded.keys():
    print(f"\nАнализ изображения: {file_name}")
    smart_photo_analyzer(file_name)

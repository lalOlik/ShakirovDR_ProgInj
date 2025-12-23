# app.py

import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import torch
import clip
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
from deep_translator import GoogleTranslator
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Умный Фото-Анализатор", layout="wide")
st.title("🖼️ Умный Фото-Анализатор")
st.markdown("Анализ изображений с помощью CLIP и BLIP")

# ==============
# Инициализация
# ==============

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.write(f"Используется устройство: `{device}`")
    
    # CLIP
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    
    # BLIP
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    blip_model.eval()
    
    translator = GoogleTranslator(source='en', target='ru')
    
    return clip_model, preprocess, blip_processor, blip_model, translator, device

clip_model, preprocess, blip_processor, blip_model, translator, device = load_models()

def load_image(image_source):
    if isinstance(image_source, str):  # URL
        response = requests.get(image_source, timeout=10)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:  # UploadedFile
        image = Image.open(image_source).convert('RGB')
    return image

def analyze_with_clip(image, categories):
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of {c}") for c in categories]).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    probs = similarity[0].cpu().numpy()
    return probs

def generate_caption(image):
    inputs = blip_processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = blip_model.generate(**inputs, max_length=70)
    caption_en = blip_processor.decode(output[0], skip_special_tokens=True)
    try:
        caption_ru = translator.translate(caption_en)
    except:
        caption_ru = caption_en
    return caption_en, caption_ru

# ==============
# Интерфейс
# ==============

input_type = st.radio("Выберите способ загрузки изображения", ("URL", "Файл"))

image = None

if input_type == "URL":
    url = st.text_input("Введите URL изображения", "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba")
    if url:
        try:
            image = load_image(url)
        except Exception as e:
            st.error(f"Ошибка загрузки: {e}")
else:
    uploaded_file = st.file_uploader("Загрузите изображение", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        try:
            image = load_image(uploaded_file)
        except Exception as e:
            st.error(f"Ошибка загрузки: {e}")

if image:
    st.image(image, caption="Загруженное изображение", width=400)

    # Категории
    default_cats = [
        "a cat", "a dog", "a person", "a car", "a nature landscape",
        "a city skyline", "food", "an interior design", "an animal",
        "technology device", "a sunset", "a beach", "mountains",
        "a forest", "a building", "art", "flowers"
    ]
    
    custom = st.checkbox("Использовать свои категории")
    if custom:
        cats_input = st.text_area("Введите категории через запятую", "modern architecture, historical building, nature scene")
        categories = [c.strip() for c in cats_input.split(",") if c.strip()]
    else:
        categories = default_cats

    if st.button("Анализировать"):
        with st.spinner("Анализ изображения..."):
            # CLIP
            probs = analyze_with_clip(image, categories)
            sorted_idx = np.argsort(probs)[::-1]

            # Таблица результатов
            st.subheader("Результаты CLIP")
            results_data = []
            for i in range(min(5, len(categories))):
                idx = sorted_idx[i]
                results_data.append({
                    "Категория": categories[idx],
                    "Вероятность (%)": f"{probs[idx]*100:.2f}"
                })
            st.table(results_data)

            top_cat = categories[sorted_idx[0]]
            top_conf = probs[sorted_idx[0]] * 100

            # BLIP
            caption_en, caption_ru = generate_caption(image)

            st.subheader("Описание от BLIP")
            st.write(f"**EN:** {caption_en}")
            st.write(f"**RU:** {caption_ru}")

            st.success(f"Лучшая категория: **{top_cat}** (уверенность: {top_conf:.2f}%)")

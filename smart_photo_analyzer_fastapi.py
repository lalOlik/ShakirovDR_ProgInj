# smart_photo_analyzer_fastapi.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union
import torch
import clip
from PIL import Image
import requests
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
from deep_translator import GoogleTranslator
import warnings
import uvicorn

warnings.filterwarnings('ignore')

# ---------------------
# Инициализация моделей
# ---------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

# BLIP
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
blip_model.eval()

# Переводчик
translator = GoogleTranslator(source='en', target='ru')

# ---------------------
# Вспомогательные функции
# ---------------------

def load_image_from_url(url: str) -> Image.Image:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка загрузки изображения по URL: {str(e)}")

def load_image_from_bytes(data: bytes) -> Image.Image:
    try:
        img = Image.open(BytesIO(data)).convert('RGB')
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка загрузки изображения из файла: {str(e)}")

def analyze_image_with_clip(image: Image.Image, categories: List[str]) -> List[dict]:
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of {c}") for c in categories]).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    probs = similarity[0].cpu().numpy()
    results = [
        {"category": categories[i], "probability": float(probs[i])}
        for i in np.argsort(probs)[::-1]
    ]
    return results

def generate_caption_blip(image: Image.Image) -> dict:
    inputs = blip_processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = blip_model.generate(**inputs, max_length=70)
    caption_en = blip_processor.decode(output[0], skip_special_tokens=True)
    try:
        caption_ru = translator.translate(caption_en)
    except:
        caption_ru = caption_en
    return {"caption_en": caption_en, "caption_ru": caption_ru}

# ---------------------
# FastAPI приложение
# ---------------------

app = FastAPI(title="Умный Фото-Анализатор", version="1.0")

# Разрешить CORS (для Streamlit или фронтенда)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    image_url: Optional[str] = None
    categories: Optional[List[str]] = None

@app.post("/analyze_url")
async def analyze_from_url(request: AnalysisRequest):
    if not request.image_url:
        raise HTTPException(status_code=400, detail="Поле image_url обязательно.")

    image = load_image_from_url(request.image_url)

    categories = request.categories or [
        "a cat", "a dog", "a person", "a car", "a nature landscape",
        "a city skyline", "food", "an interior design", "an animal",
        "technology device", "a sunset", "a beach", "mountains",
        "a forest", "a building", "art", "flowers"
    ]

    clip_results = analyze_image_with_clip(image, categories)
    caption = generate_caption_blip(image)

    return {
        "top_category": clip_results[0]["category"],
        "confidence": clip_results[0]["probability"] * 100,
        "clip_results": clip_results,
        "caption": caption
    }

@app.post("/analyze_file")
async def analyze_from_file(
    file: UploadFile = File(...),
    categories: Optional[str] = None  # передавать как строку, разделённую запятыми
):
    contents = await file.read()
    image = load_image_from_bytes(contents)

    cat_list = [c.strip() for c in categories.split(",")] if categories else [
        "a cat", "a dog", "a person", "a car", "a nature landscape",
        "a city skyline", "food", "an interior design", "an animal",
        "technology device", "a sunset", "a beach", "mountains",
        "a forest", "a building", "art", "flowers"
    ]

    clip_results = analyze_image_with_clip(image, cat_list)
    caption = generate_caption_blip(image)

    return {
        "top_category": clip_results[0]["category"],
        "confidence": clip_results[0]["probability"] * 100,
        "clip_results": clip_results,
        "caption": caption
    }

@app.get("/")
def root():
    return {"message": "Умный Фото-Анализатор API работает! Используйте /analyze_url или /analyze_file"}

import sys
import io
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

# ============================================================================
# ФИКСТУРЫ И МОКИ (Запускаются ДО импорта app.py)
# ============================================================================

@pytest.fixture(autouse=True)
def mock_ml_dependencies():
    """Фикстура автоматически подменяет тяжелые ML-модели и переводчик,
    чтобы тесты в CI/CD проходили мгновенно и без интернета."""
    
    with patch('clip.load') as mock_clip, \
         patch('transformers.BlipProcessor.from_pretrained') as mock_blip_proc, \
         patch('transformers.BlipForConditionalGeneration.from_pretrained') as mock_blip_model, \
         patch('deep_translator.GoogleTranslator') as mock_trans:
        
        # Мокаем CLIP (возвращает фейковую модель и препроцессор)
        mock_model = MagicMock()
        # Симулируем поведение векторов схожести для CLIP
        mock_model.encode_image.return_value = MagicMock()
        mock_model.encode_text.return_value = MagicMock()
        mock_clip.return_value = (mock_model, MagicMock())
        
        # Мокаем BLIP
        mock_blip_proc.return_value = MagicMock()
        mock_dummy_output = [MagicMock()]
        mock_blip_model.return_value.to.return_value.generate.return_value = mock_dummy_output
        mock_blip_proc.return_value.decode.return_value = "a beautiful cat"
        
        # Мокаем переводчик
        mock_trans.return_value.translate.return_value = "красивый кот"
        
        yield

@pytest.fixture
def app_module():
    """Безопасно импортирует app.py внутри тестового контекста."""
    # Удаляем из кэша импортов, если он там был, для чистоты тестов
    if 'app' in sys.modules:
        del sys.modules['app']
    import app
    return app

@pytest.fixture
def sample_image():
    """Создает фейковое изображение в памяти для тестов."""
    img = Image.new('RGB', (100, 100), color='red')
    return img

# ============================================================================
# ТЕСТЫ БИЗНЕС-ЛОГИКИ
# ============================================================================

def test_load_image_from_file(app_module, sample_image):
    """Проверяет загрузку изображения, переданного как файл (BytesIO)."""
    img_byte_arr = io.BytesIO()
    sample_image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    result = app_module.load_image(img_byte_arr)
    
    assert isinstance(result, Image.Image)
    assert result.size == (100, 100)

@patch('requests.get')
def test_load_image_from_url(mock_get, app_module, sample_image):
    """Проверяет загрузку изображения по URL с моком сетевого запроса."""
    img_byte_arr = io.BytesIO()
    sample_image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    # Настраиваем фейковый ответ от сервера
    mock_response = MagicMock()
    mock_response.content = img_byte_arr.read()
    mock_get.return_value = mock_response
    
    result = app_module.load_image("https://fake-url.com/image.jpg")
    
    assert isinstance(result, Image.Image)
    mock_get.assert_called_once_with("https://fake-url.com/image.jpg", timeout=10)

def test_generate_caption(app_module, sample_image):
    """Проверяет работу генерации описания и переводчика (через моки)."""
    caption_en, caption_ru = app_module.generate_caption(sample_image)
    
    assert caption_en == "a beautiful cat"
    assert caption_ru == "красивый кот"

# ============================================================================
# ТЕСТ ИНТЕРФЕЙСА (STREAMLIT APPTEST)
# ============================================================================

def test_streamlit_ui_render():
    """Проверяет, что интерфейс приложения Streamlit успешно отрисовывается
    и не падает с критической ошибкой при старте."""
    from streamlit.testing.v1 import AppTest
    
    # Запускаем симуляцию приложения app.py
    at = AppTest.from_file("app.py").run()
    
    # Проверяем наличие заголовка и элементов управления
    assert not at.exception
    assert len(at.title) > 0
    assert at.title[0].value == "🖼️ Умный Фото-Анализатор"
    
    # Проверяем, что радио-кнопка выбора типа загрузки присутствует
    assert at.radio
    assert at.radio[0].value == "URL"

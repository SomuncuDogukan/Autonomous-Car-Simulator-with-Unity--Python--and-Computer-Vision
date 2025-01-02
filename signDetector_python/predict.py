import logging
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms
from train import TrafficSignNet  # Model sınıfı

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# FastAPI uygulaması
app = FastAPI()

# Modeli yükle
model = TrafficSignNet()
model.load_state_dict(torch.load("traffic_sign_model.pth"))
model.eval()

# Güven eşiği (threshold) belirleme
CONFIDENCE_THRESHOLD = 0.70  # Modelin tahmininin en az %70 güvenle yapılmasını istiyoruz

def predict_image(image_bytes):
    try:
        image = Image.open(BytesIO(image_bytes))
        logger.info("Görsel başarıyla yüklendi")

        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        image = transform(image).unsqueeze(0)
        logger.info("Görsel başarıyla dönüştürüldü")

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted].item()  # Güven hesaplama

        # Güven eşiği ile kontrol
        if confidence < CONFIDENCE_THRESHOLD:
            logger.info(f"Güven düşük: {confidence}. Sonuç geçersiz.")
            return "Unknown"  # Güven düşükse, geçersiz sonuç döndür

        # Eğer güven yeterliyse, tahmin edilen sınıfı döndür
        logger.info(f"Güven: {confidence}, Model tahmini: {predicted.item()}")
        return 20 if predicted.item() == 0 else 30

    except Exception as e:
        logger.error(f"Predict fonksiyonunda hata: {e}")
        raise e


@app.post("/predict/")
async def predict(file: UploadFile):
    try:
        logger.info(f"Gelen dosya: {file.filename}")
        image_bytes = await file.read()
        speed_limit = predict_image(image_bytes)
        logger.info(f"Tahmin edilen hız sınırı: {speed_limit}")

        return JSONResponse(content={"speed_limit": speed_limit})
    except Exception as e:
        logger.error(f"API hata: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

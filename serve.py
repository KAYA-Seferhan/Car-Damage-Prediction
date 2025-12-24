import json
import torch
import gradio as gr
from PIL import Image
from torchvision import transforms
from model import CNNCarDamage

css = """
footer {display: none !important;}
"""

MODEL_PATH = "car_damage_cnn.pth"
META_PATH  = "car_damage_meta.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

THRESHOLD = 0.39

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

IMG_SIZE = meta["img_size"]
pos_idx = int(meta["positive_class_idx"])
idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()}
pos_name = meta["positive_class_name"]

tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

model = CNNCarDamage().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def predict(image: Image.Image):
    x = tfm(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logit = model(x)
        p_pos = torch.sigmoid(logit).item()

    label_tr = "Hasarlı" if p_pos >= THRESHOLD else "Hasarsız"
    return {
        "Threshold": THRESHOLD,
        "Pozitif sınıf (hasar) adı": pos_name,
        "Hasarlı olasılığı": float(p_pos),
        "Hasarsız olasılığı": float(1 - p_pos),
        "Tahmin": label_tr
    }

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.JSON(),
    title="Araç Hasar Tespiti (CNN)",
    description="Görüntü yükle → hasarlı/hasarsız tahmini + olasılıklar."
)

if __name__ == "__main__":
    demo.launch(share=True, css=css)

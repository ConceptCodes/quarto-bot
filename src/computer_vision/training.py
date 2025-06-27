from ultralytics import YOLO

print("Loading the pretrained model...")
model = YOLO("data/yolo11s.pt")

results = model.train(
    data="src/config/training_config.yaml",
    imgsz=416,
    epochs=50,
    batch=16,
    name="quarto_model2",
    plots=True,
    amp=True,
)

from ultralytics import YOLO

print("Loading the pretrained model...")
model = YOLO("runs/detect/quarto_model2/weights/best.pt")

print("Running inference on the image...")
results = model.predict(source="IMG_5003.png", conf=0.25, show=True, save=True)
print("Inference completed. Results saved to runs/detect/quarto_model/predict")

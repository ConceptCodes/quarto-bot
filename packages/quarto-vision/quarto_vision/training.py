from ultralytics import YOLO

print("Loading the pretrained model...")
model = YOLO("runs/detect/quarto_model_optimized_v2/weights/last.pt")

results = model.train(
    data="src/config/training_config.yaml",
    imgsz=640,  # Increased for better detection of small pieces
    epochs=50,  # Fewer epochs for fine-tuning
    batch=8,  # Smaller batch size for small dataset
    lr0=0.0005,  # Even lower learning rate for fine-tuning
    weight_decay=0.0005,  # Regularization for small dataset
    patience=20,  # Early stopping patience
    augment=True,  # Enable data augmentation
    copy_paste=0.3,  # Copy-paste augmentation
    mixup=0.2,  # Mixup augmentation
    mosaic=1.0,  # Mosaic augmentation
    hsv_h=0.015,  # HSV hue augmentation
    hsv_s=0.7,  # HSV saturation augmentation
    hsv_v=0.4,  # HSV value augmentation
    degrees=10.0,  # Rotation augmentation
    translate=0.1,  # Translation augmentation
    scale=0.5,  # Scale augmentation
    fliplr=0.5,  # Horizontal flip
    name="quarto_model_optimized_v2",
    plots=True,
    amp=True,
    save_period=10,  # Save checkpoint every 10 epochs
    device="mps", # Use Metal Performance Shaders for Apple Silicon Macs
)

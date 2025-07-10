from ultralytics import YOLO

print("Loading the pretrained model...")
model = YOLO("data/yolo11s.pt")

results = model.train(
    data="src/config/training_config.yaml",
    imgsz=640,  # Increased for better detection of small pieces
    epochs=100,  # More epochs for small dataset
    batch=8,  # Smaller batch size for small dataset
    lr0=0.001,  # Lower learning rate for better convergence
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
    name="quarto_model_optimized",
    plots=True,
    amp=True,
    save_period=10,  # Save checkpoint every 10 epochs
    device="mps", # Use Metal Performance Shaders for Apple Silicon Macs
)

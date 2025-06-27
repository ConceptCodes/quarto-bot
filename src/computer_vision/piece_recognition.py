from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser(description="Run YOLO inference on an image or video.")
parser.add_argument(
    "source",
    type=str,
    help="Path to image/video file or camera index (e.g., 0 for webcam)",
)
parser.add_argument("--conf", type=float, default=0.75, help="Confidence threshold")
args = parser.parse_args()

print("Loading the pretrained model...")
model = YOLO("runs/detect/quarto_model2/weights/best.pt")

print(f"Running inference on the source: {args.source}")
source = int(args.source) if args.source.isdigit() else args.source
results = model.predict(source=source, conf=args.conf, show=True, save=True)

if not str(args.source).isdigit():
    print("Inference completed. Results saved to runs/detect/quarto_model2/predict")

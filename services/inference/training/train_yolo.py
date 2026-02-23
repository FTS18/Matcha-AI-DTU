import os
import argparse
from ultralytics import YOLO

def train_custom_yolo(data_yaml, model_size="s", epochs=50, imgsz=640):
    """
    Fine-tune YOLO on your custom frame-by-frame dataset.
    
    Args:
        data_yaml (str): Path to data.yaml file (defines paths and classes)
        model_size (str): 'n', 's', 'm', 'l', 'x' (default 's' for small)
        epochs (int): Number of training epochs
        imgsz (int): Image size for training
    """
    model_name = f"yolov8{model_size}.pt"
    print(f"🚀 Loading base model: {model_name}")
    
    # Load a pretrained model
    model = YOLO(model_name)
    
    # Train the model
    print(f"🏋️ Starting training with {epochs} epochs on {data_yaml}...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        plots=True,
        save=True,
        project="matcha_training",
        name="soccer_custom"
    )
    
    if results and results.save_dir:
        print(f"✅ Training complete! Best weights saved to: {results.save_dir}/weights/best.pt")
        print("👉 Copy these weights to services/inference/ and update analysis.py to use them.")
    else:
        print("✅ Training complete! Check the matcha_training/ directory for weights.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLO for Soccer")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--size", type=str, default="s", help="Model size (n, s, m, l, x)")
    
    args = parser.parse_args()
    
    # Ensure structure exists
    os.makedirs("matcha_training", exist_ok=True)
    
    train_custom_yolo(args.data, args.size, args.epochs, args.imgsz)

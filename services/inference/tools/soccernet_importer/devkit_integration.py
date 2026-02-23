import os
import torch
import numpy as np
from PIL import Image
from SoccerNet.Evaluation.ActionSpotting import evaluate
from SoccerNet.Features.Extractor import VideoFeatureExtractor

def extract_features_official(video_path, model_path="models/soccernet/resnet152.pth"):
    """
    Extracts features using the official SoccerNet DevKit method.
    This corresponds to the 'Phase 1' of their feature extraction pipeline.
    """
    print(f"🎬 Processing video: {video_path}")
    
    # Initialize implementation from DevKit
    # Note: This requires the ResNet152 weights from SoccerNet
    extractor = VideoFeatureExtractor(
        video_path=video_path,
        feature_path=None, # In-memory
        backbone="resnet152",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Extract features
    features = extractor.extract_features()
    print(f"✅ Extracted {features.shape} features")
    return features

def evaluate_predictions(results_json, labels_json):
    """
    Evaluate your model's JSON output against official SoccerNet labels.
    Uses the official mAP (mean Average Precision) metric.
    """
    print("📊 Evaluating Match Performance...")
    results = evaluate(
        LabelsPath=labels_json,
        PredictionsPath=results_json,
        split="test",
        prediction_file="Predictions-ActionSpotting.json"
    )
    
    print(f"🏆 Average Precision: {results['mAP']}")
    return results

if __name__ == "__main__":
    print("This tool integrates with the official SoccerNet DevKit.")
    print("See services/inference/app/core/soccernet_detector.py for current implementation.")

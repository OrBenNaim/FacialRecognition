import os
from src.face_recognition_model import FaceRecognition


def run_test_pipeline(test_model: FaceRecognition):
    """Simple test pipeline."""

    print("🔄 Loading trained model...")
    if not os.path.exists("best_model.pth"):
        print("❌ No trained model found! Please run training first.")
        return None

    test_model.load_best_model("best_model.pth")

    print("🎯 Running test evaluation...")
    test_results = test_model.run_test_evaluation()

    test_model.writer.close()

    print("✅ Testing completed!")
    return test_results

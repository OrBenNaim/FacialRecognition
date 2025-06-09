from src.face_recognition_model import FaceRecognition


def run_training_pipeline(model: FaceRecognition):
    """Simple training pipeline."""

    print("⚙️  Setting up training...")
    model.reset_exp_attr()

    # Simple experiment name
    exp_name = "siamese_enhanced_aug"

    print("🎯 Training model...")
    model.run_complete_experiment(
        learning_rate=6e-5,
        batch_size=32,
        epochs=50,
        use_improved_arch=False,  # Base architecture
        exp_name=exp_name
    )

    print("✅ Training completed!")
    return model

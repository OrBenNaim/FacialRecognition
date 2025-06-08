import os
import sys
import time
import argparse

from src.main_training import run_training_pipeline, verify_prerequisites as verify_train_prereq
from src.main_testing import run_test_pipeline, verify_test_prerequisites


def print_header():
    """Print welcome header."""
    print("🚀" + "=" * 70 + "🚀")
    print("    SIAMESE NEURAL NETWORK - COMPLETE PIPELINE")
    print("    One-Shot Facial Recognition Implementation")
    print("🚀" + "=" * 70 + "🚀")


def print_phase_separator(phase_name):
    """Print phase separator."""
    print("\n" + "🔄" * 20)
    print(f"    {phase_name}")
    print("🔄" * 20 + "\n")


def verify_prerequisites():
    """Verify all prerequisites before starting."""
    print("🔍 Verifying prerequisites...")

    required_files = [
        "./DATA/LFW-a",
        "./DATA/pairsDevTrain.txt",
        "./DATA/pairsDevTest.txt"
    ]

    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False

    # Create necessary directories
    os.makedirs("tensorboard_logs", exist_ok=True)
    os.makedirs("test_results", exist_ok=True)
    os.makedirs("images", exist_ok=True)

    print("✅ All prerequisites satisfied")
    return True


def run_training_phase():
    """Execute training phase."""
    print_phase_separator("PHASE 1: TRAINING")

    try:
        if not verify_train_prereq():
            print("❌ Training prerequisites not met")
            return False

        print("🎯 Starting training phase...")
        start_time = time.time()

        trained_model = run_training_pipeline()

        training_time = time.time() - start_time
        print(f"\n✅ Training phase completed in {training_time / 60:.2f} minutes")

        # Verify model was saved
        if os.path.exists("best_model.pth"):
            model_size = os.path.getsize("best_model.pth") / (1024 * 1024)
            print(f"📦 Model saved: best_model.pth ({model_size:.2f} MB)")
            return True
        else:
            print("❌ Model not saved properly")
            return False

    except Exception as e:
        print(f"❌ Training phase failed: {e}")
        return False


def run_testing_phase():
    """Execute testing phase."""
    print_phase_separator("PHASE 2: TESTING")

    try:
        if not verify_test_prerequisites():
            print("❌ Testing prerequisites not met")
            return False

        print("🎯 Starting testing phase...")
        start_time = time.time()

        test_results = run_test_pipeline()

        testing_time = time.time() - start_time
        print(f"\n✅ Testing phase completed in {testing_time / 60:.2f} minutes")

        return test_results

    except Exception as e:
        print(f"❌ Testing phase failed: {e}")
        return None


def generate_final_summary(test_results, total_time):
    """Generate final pipeline summary."""
    print("\n" + "🏆" + "=" * 70 + "🏆")
    print("    COMPLETE PIPELINE SUMMARY")
    print("🏆" + "=" * 70 + "🏆")

    print(f"\n⏱️  TIMING:")
    print(f"   Total Pipeline Time: {total_time / 60:.2f} minutes")

    if test_results:
        print(f"\n📊 FINAL TEST RESULTS:")
        print(f"   Test Accuracy:      {test_results['accuracy']:.4f} ({test_results['accuracy'] * 100:.2f}%)")
        print(f"   True Positive Rate: {test_results['tpr']:.4f} ({test_results['tpr'] * 100:.2f}%)")
        print(f"   True Negative Rate: {test_results['tnr']:.4f} ({test_results['tnr'] * 100:.2f}%)")
        print(f"   F1 Score:          {test_results['f1_score']:.4f}")
        print(f"   AUC Score:         {test_results['auc']:.4f}")

        # Performance assessment
        validation_expected = 0.807  # Your enhanced augmentation result
        performance_drop = validation_expected - test_results['accuracy']

        print(f"\n📈 GENERALIZATION ANALYSIS:")
        print(f"   Expected Validation: {validation_expected:.4f}")
        print(f"   Actual Test:        {test_results['accuracy']:.4f}")
        print(f"   Performance Drop:   {performance_drop:.4f} ({performance_drop * 100:.2f}%)")

        if performance_drop < 0.03:
            assessment = "EXCELLENT ✨"
        elif performance_drop < 0.07:
            assessment = "GOOD ✅"
        elif performance_drop < 0.12:
            assessment = "MODERATE 📊"
        else:
            assessment = "NEEDS IMPROVEMENT ⚠️"

        print(f"   Assessment:         {assessment}")

    print(f"\n📁 OUTPUTS GENERATED:")
    print(f"   Model File:         best_model.pth")
    print(f"   TensorBoard Logs:   tensorboard_logs/")
    print(f"   Test Report:        test_results/final_test_evaluation_report.txt")
    print(f"   Dataset Analysis:   images/")

    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. View training curves:    tensorboard --logdir=tensorboard_logs")
    print(f"   2. Read detailed report:    test_results/final_test_evaluation_report.txt")
    print(f"   3. Analyze misclassifications in TensorBoard")

    print("\n" + "🎉" + "=" * 70 + "🎉")
    print("    SIAMESE NETWORK PIPELINE COMPLETED SUCCESSFULLY!")
    print("🎉" + "=" * 70 + "🎉")


def main():
    """Main pipeline orchestrator."""
    parser = argparse.ArgumentParser(description='Siamese Network Complete Pipeline')
    parser.add_argument('--train-only', action='store_true', help='Run only training phase')
    parser.add_argument('--test-only', action='store_true', help='Run only testing phase')
    parser.add_argument('--skip-training', action='store_true', help='Skip training if model exists')
    parser.add_argument('--clear-cache', action='store_true', help='Clear data cache before training')
    parser.add_argument('--cache-info', action='store_true', help='Show cache information and exit')

    args = parser.parse_args()

    # Handle cache info request
    if args.cache_info:
        from src.face_recognition_model import FaceRecognition
        from src.constants import OPTIMIZED_IMG_SHAPE

        model = FaceRecognition(input_shape=OPTIMIZED_IMG_SHAPE)
        cache_info = model.get_cache_info()

        if cache_info:
            print("📂 Data Cache Information:")
            print(f"   Created: {cache_info.get('creation_time', 'Unknown')}")
            print(f"   Training pairs: {cache_info.get('train_pairs_count', 0):,}")
            print(f"   Validation pairs: {cache_info.get('val_pairs_count', 0):,}")
            print(f"   Unique people: {cache_info.get('unique_people', 0):,}")
            print(f"   Input shape: {cache_info.get('input_shape', 'Unknown')}")
            print(f"   Validation split: {cache_info.get('validation_split', 'Unknown')}")
        else:
            print("📂 No data cache found")
        return

    # Handle cache clearing
    if args.clear_cache:
        from src.face_recognition_model import FaceRecognition
        from src.constants import OPTIMIZED_IMG_SHAPE

        model = FaceRecognition(input_shape=OPTIMIZED_IMG_SHAPE)
        model.clear_data_cache()

        if not (args.train_only or args.test_only):
            print("Cache cleared. Add --train-only or other options to continue.")
            return

    # Print header
    print_header()

    # Record start time
    pipeline_start_time = time.time()

    # Verify prerequisites
    if not verify_prerequisites():
        print("❌ Prerequisites not met. Please check your data setup.")
        sys.exit(1)

    # Phase execution logic
    training_success = False
    test_results = None

    # Phase 1: Training
    if not args.test_only:
        if args.skip_training and os.path.exists("best_model.pth"):
            print("⏭️  Skipping training - model already exists")
            training_success = True
        else:
            training_success = run_training_phase()

        if not training_success:
            print("❌ Training failed. Cannot proceed to testing.")
            sys.exit(1)
    else:
        # Check if model exists for test-only mode
        if not os.path.exists("best_model.pth"):
            print("❌ No trained model found. Please run training first.")
            sys.exit(1)
        training_success = True

    # Phase 2: Testing
    if not args.train_only and training_success:
        test_results = run_testing_phase()

        if test_results is None:
            print("❌ Testing failed.")
            sys.exit(1)

    # Generate final summary
    total_time = time.time() - pipeline_start_time
    generate_final_summary(test_results, total_time)


if __name__ == '__main__':
    main()
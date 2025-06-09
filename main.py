import os
import time

from torch.utils.tensorboard import SummaryWriter

from src.constants import OPTIMIZED_IMG_SHAPE, VALIDATION_SPLIT, SAVED_CACHE_DATASET_PATH, SAVED_LOGS_PATH
from src.face_recognition_model import FaceRecognition
from src.main_training import run_training_pipeline
from src.main_testing import run_test_pipeline
from src.utils import print_header, verify_prerequisites


def run_training(model: FaceRecognition):
    """Run training phase."""
    print("\n🎯 PHASE 1: TRAINING")
    print("=" * 30)

    try:
        start_time = time.time()
        _ = run_training_pipeline(model)
        training_time = time.time() - start_time

        print(f"\n✅ Training completed in {training_time / 60:.1f} minutes")

        # Check if the model was saved
        if os.path.exists("best_model.pth"):
            model_size = os.path.getsize("best_model.pth") / (1024 * 1024)
            print(f"💾 Model saved: best_model.pth ({model_size:.1f} MB)")
            return True
        else:
            print("❌ Model not saved properly")
            return False

    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False


def run_testing(test_model: FaceRecognition):
    """Run the testing phase."""
    print("\n🎯 PHASE 2: TESTING")
    print("=" * 30)

    try:
        start_time = time.time()
        test_results = run_test_pipeline(test_model)
        testing_time = time.time() - start_time

        print(f"\n✅ Testing completed in {testing_time / 60:.1f} minutes")
        return test_results

    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return None


def show_final_results(test_results, total_time):
    """Show the final summary."""
    print("\n🏆" + "=" * 50 + "🏆")
    print("           FINAL RESULTS")
    print("🏆" + "=" * 50 + "🏆")

    print(f"\n⏱️  Total Time: {total_time / 60:.1f} minutes")

    if test_results:
        print(f"\n📊 Test Performance:")
        print(f"   Accuracy:      {test_results['accuracy']:.3f} ({test_results['accuracy'] * 100:.1f}%)")
        print(f"   Same Person:   {test_results['tpr']:.3f} ({test_results['tpr'] * 100:.1f}%)")
        print(f"   Different:     {test_results['tnr']:.3f} ({test_results['tnr'] * 100:.1f}%)")
        print(f"   F1 Score:      {test_results['f1_score']:.3f}")
        print(f"   AUC:          {test_results['auc']:.3f}")

        # Simple assessment
        if test_results['accuracy'] > 0.75:
            print("\n🎉 Great results!")
        elif test_results['accuracy'] > 0.70:
            print("\n✅ Good results!")
        else:
            print("\n📊 Reasonable results")

    print(f"\n📁 Generated Files:")
    print(f"   Model:         best_model.pth")
    print(f"   TensorBoard:   tensorboard_logs/")
    print(f"   Test Report:   test_results/")

    print(f"\n🎯 View Results:")
    print(f"   tensorboard --logdir=tensorboard_logs")

    print("\n🎉 Pipeline completed successfully! 🎉")


def main():
    """Simple main pipeline."""
    print_header()  # print welcome header

    # Check prerequisites
    if not verify_prerequisites():
        print("❌ Please check your data setup and try again.")
        return

    print("📋 Initializing model...")
    model = FaceRecognition(input_shape=OPTIMIZED_IMG_SHAPE)

    # Try to load from the cache first
    if model.load_dataset_from_cache(SAVED_CACHE_DATASET_PATH):
        print("⚡ Training data loaded from cache (fast!)")

    else:
        print("🔄 Loading and preprocessing data (this may take a few minutes)...")
        model.load_lfw_dataset(VALIDATION_SPLIT)

        # Save for future use
        model.save_dataset_to_cache(SAVED_CACHE_DATASET_PATH)

    print("📊 Analyzing dataset...")
    analysis_log_dir = os.path.join(SAVED_LOGS_PATH, "dataset_analysis")
    os.makedirs(analysis_log_dir, exist_ok=True)

    model.tensorboard_log_dir = analysis_log_dir
    model.writer = SummaryWriter(log_dir=analysis_log_dir)
    model.log_dataset_analysis_to_tensorboard()
    model.writer.close()

    # Phase 1: Training
    print("\nStarting complete pipeline...")
    start_time = time.time()

    training_success = run_training(model)

    if not training_success:
        print("❌ Training failed. Cannot proceed to testing.")
        return

    use_test = input("\nDo you want to run testing? (Press y/n)\n").lower()

    if use_test != ('y' or 'n'):
        raise Exception("\nPlease press on 'y' or 'n' only.\n")

    if use_test == 'y':
        # Phase 2: Testing
        test_results = run_testing(model)

        if test_results is None:
            print("❌ Testing failed.")
            return

        total_time = time.time() - start_time
        show_final_results(test_results, total_time)

if __name__ == '__main__':
    main()
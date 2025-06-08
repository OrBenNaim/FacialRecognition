import os

from torch.utils.tensorboard import SummaryWriter

from src.constants import OPTIMIZED_IMG_SHAPE, DATA_FOLDER_PATH, TEST_FILE_PATH
from src.face_recognition_model import FaceRecognition


def run_test_pipeline():
    """
    Complete the test evaluation pipeline.
    
    Steps:
    1. Load test dataset (completely separate from training)
    2. Load pre-trained model (no retraining!)
    3. Evaluate on test set
    4. Generate a comprehensive test report
    """
    
    print("🧪 SIAMESE NETWORK TEST PIPELINE")
    print("=" * 60)
    print("Testing trained model on completely unseen test data...")
    
    # ========= Step 1: Initialize Test Model =========
    print("\n📋 Step 1: Initializing test model...")
    test_model = FaceRecognition(input_shape=OPTIMIZED_IMG_SHAPE)
    print("✅ Test model initialized")
    
    # ========= Step 2: Load Test Dataset =========
    print(f"\n📁 Step 2: Loading test dataset...")
    test_model.load_lfw_dataset(DATA_FOLDER_PATH, TEST_FILE_PATH)  # No validation split for test
    print("✅ Test dataset loaded")
    
    # ========= Step 3: Setup Test Logging =========
    print(f"\n📊 Step 3: Setting up test logging...")
    test_log_dir = os.path.join("tensorboard_logs", "test_evaluation")
    os.makedirs(test_log_dir, exist_ok=True)
    
    test_model.tensorboard_log_dir = test_log_dir
    test_model.writer = SummaryWriter(log_dir=test_log_dir)
    print("✅ Test logging configured")
    
    # ========= Step 4: Analyze Test Dataset =========
    print(f"\n📈 Step 4: Analyzing test dataset...")
    test_model.log_test_dataset_analysis()
    print("✅ Test dataset analysis completed")
    
    # ========= Step 5: Load Pre-trained Model =========
    print(f"\n🔄 Step 5: Loading pre-trained model...")
    model_path = "best_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model file '{model_path}' not found!")
        print("Please run training first: python main_training.py")
        return None
    
    test_model.load_best_model(model_path)
    print("✅ Pre-trained model loaded successfully")
    
    # ========= Step 6: Run Test Evaluation =========
    print(f"\n🎯 Step 6: Running test evaluation...")
    print("=" * 40)
    
    try:
        test_results = test_model.run_test_evaluation()
        print("=" * 40)
        print("✅ Test evaluation completed!")
        
    except Exception as e:
        print(f"❌ Test evaluation failed: {e}")
        raise
    
    # ========= Step 7: Generate Test Report =========
    print(f"\n📋 Step 7: Generating test report...")
    test_model.generate_test_report(test_results)
    print("✅ Test report generated")
    
    # ========= Step 8: Display Final Results =========
    print(f"\n🏆 Step 8: Final Test Results")
    print("=" * 50)
    print(f"Test Accuracy:        {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.2f}%)")
    print(f"True Positive Rate:   {test_results['tpr']:.4f} ({test_results['tpr']*100:.2f}%)")
    print(f"True Negative Rate:   {test_results['tnr']:.4f} ({test_results['tnr']*100:.2f}%)")
    print(f"F1 Score:            {test_results['f1_score']:.4f}")
    print(f"AUC:                 {test_results['auc']:.4f}")
    print(f"Average Precision:   {test_results['average_precision']:.4f}")
    
    # ========= Step 9: Performance Analysis =========
    print(f"\n📊 Step 9: Performance Analysis")
    print("=" * 40)
    
    # Compare to expected validation results
    validation_accuracy = 0.807  # Your enhanced augmentation result
    performance_drop = validation_accuracy - test_results['accuracy']
    
    print(f"Expected Validation:  {validation_accuracy:.4f}")
    print(f"Actual Test:         {test_results['accuracy']:.4f}")
    print(f"Performance Drop:    {performance_drop:.4f} ({performance_drop*100:.2f}%)")
    
    if performance_drop > 0.05:
        print("⚠️  Significant performance drop - indicates overfitting")
    elif performance_drop < 0.02:
        print("✅ Excellent generalization - minimal performance drop")
    else:
        print("📊 Good generalization - acceptable performance drop")
    
    # ========= Step 10: Model Behavior Analysis =========
    print(f"\n🔍 Step 10: Model Behavior Analysis")
    print("=" * 40)
    print(f"Same-person detection: {test_results['tpr']*100:.1f}% success rate")
    print(f"Different-person detection: {test_results['tnr']*100:.1f}% success rate")
    
    if test_results['tpr'] > test_results['tnr']:
        bias_diff = test_results['tpr'] - test_results['tnr']
        print(f"📋 Model bias: {bias_diff*100:.1f}% better at same-person detection")
    else:
        bias_diff = test_results['tnr'] - test_results['tpr']
        print(f"📋 Model bias: {bias_diff*100:.1f}% better at different-person detection")
    
    # ========= Step 11: Wrap Up =========
    print(f"\n🎯 View detailed results:")
    print(f"TensorBoard: tensorboard --logdir={test_log_dir}")
    print(f"Test Report: test_results/final_test_evaluation_report.txt")
    
    test_model.writer.close()
    
    print(f"\n🏁 TEST PIPELINE COMPLETED")
    print("=" * 60)
    
    return test_results

def verify_test_prerequisites():
    """Verify test prerequisites."""
    print("🔍 Verifying test prerequisites...")
    
    # Check test data
    if not os.path.exists(DATA_FOLDER_PATH):
        print(f"❌ Data folder missing: {DATA_FOLDER_PATH}")
        return False
        
    if not os.path.exists(TEST_FILE_PATH):
        print(f"❌ Test file missing: {TEST_FILE_PATH}")
        return False
    
    # Check for a trained model
    if not os.path.exists("best_model.pth"):
        print(f"❌ Trained model missing: best_model.pth")
        print(f"Please run training first: python main_training.py")
        return False
    
    # Create directories
    os.makedirs("test_results", exist_ok=True)
    
    print("✅ Test prerequisites verified")
    return True

if __name__ == '__main__':
    if not verify_test_prerequisites():
        print("❌ Test prerequisites not met.")
        exit(1)
    
    # Run test pipeline
    results = run_test_pipeline()
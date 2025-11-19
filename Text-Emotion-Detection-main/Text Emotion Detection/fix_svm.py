# Quick Fix for SVM Model - Run this in your notebook
# Copy and paste this entire cell into your notebook and run it

print("🔧 Quick SVM Fix - Training SVM with probability=True...")

# Re-import libraries just in case
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import joblib
import os

# Retrain SVM with probability estimates
print("🚀 Training SVM with probability estimates...")
pipe_svm_fixed = Pipeline(steps=[
    ('cv', CountVectorizer()),
    ('svc', SVC(kernel='rbf', C=10, probability=True))
])

# Train the model
pipe_svm_fixed.fit(x_train, y_train)
svm_accuracy = pipe_svm_fixed.score(x_test, y_test)
print(f"✅ SVM training complete! Accuracy: {svm_accuracy:.4f}")

# Test probability prediction
test_text = ["I am very happy today!"]
test_pred = pipe_svm_fixed.predict(test_text)
test_proba = pipe_svm_fixed.predict_proba(test_text)
print(f"✅ Probability test successful!")
print(f"   Prediction: {test_pred[0]}")
print(f"   Classes: {pipe_svm_fixed.classes_}")
print(f"   Probabilities shape: {test_proba.shape}")

# Save the fixed model
os.makedirs("model", exist_ok=True)
joblib.dump(pipe_svm_fixed, "model/svm.pkl")
print("✅ Fixed SVM model saved!")

# Update the global variable
pipe_svm = pipe_svm_fixed
print("✅ Global pipe_svm variable updated!")

print("\n🎉 SVM fix complete! Now restart your Streamlit app.")
print("   The SVM model now supports probability predictions!")

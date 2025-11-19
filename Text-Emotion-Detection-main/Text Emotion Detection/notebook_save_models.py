# ==============================================
# COPY THIS ENTIRE CELL TO YOUR JUPYTER NOTEBOOK
# Run this after training all your models
# ==============================================

import joblib
import os

print("💾 Saving all trained models for Streamlit app...")

# Create model directory
os.makedirs("model", exist_ok=True)

models_saved = 0
errors = []

# Save Logistic Regression
try:
    joblib.dump(pipe_lr, "model/logreg.pkl")
    accuracy_lr = pipe_lr.score(x_test, y_test)
    print(f"✅ Logistic Regression saved (Accuracy: {accuracy_lr:.4f})")
    models_saved += 1
except Exception as e:
    errors.append(f"❌ Logistic Regression (pipe_lr): {e}")

# Save SVM
try:
    joblib.dump(pipe_svm, "model/svm.pkl")
    accuracy_svm = pipe_svm.score(x_test, y_test)
    print(f"✅ SVM saved (Accuracy: {accuracy_svm:.4f})")
    models_saved += 1
except Exception as e:
    errors.append(f"❌ SVM (pipe_svm): {e}")

# Save Random Forest
try:
    joblib.dump(pipe_rf, "model/rf.pkl")
    accuracy_rf = pipe_rf.score(x_test, y_test)
    print(f"✅ Random Forest saved (Accuracy: {accuracy_rf:.4f})")
    models_saved += 1
except Exception as e:
    errors.append(f"❌ Random Forest (pipe_rf): {e}")

# Save LSTM model
try:
    model_lstm.save("model/lstm_model.h5")
    print(f"✅ LSTM saved (Accuracy: {lstm_acc:.4f})")
    models_saved += 1
except Exception as e:
    errors.append(f"❌ LSTM (model_lstm): {e}")

# Save Label Encoder (needed for LSTM)
try:
    joblib.dump(le, "model/label_encoder.pkl")
    print("✅ Label encoder saved")
except Exception as e:
    errors.append(f"❌ Label encoder (le): {e}")

# Save Tokenizer (needed for LSTM)
try:
    joblib.dump(tokenizer, "model/tokenizer.pkl")
    print("✅ Tokenizer saved")
except Exception as e:
    errors.append(f"❌ Tokenizer: {e}")

# Summary
print(f"\n📊 SUMMARY:")
print(f"✅ {models_saved} models saved successfully")

if errors:
    print(f"❌ {len(errors)} errors occurred:")
    for error in errors:
        print(f"   {error}")
    print("\n💡 Make sure you have trained all models in your notebook before running this cell")
else:
    print("🎉 All models saved! Your Streamlit app now has access to:")
    print("   • Logistic Regression")
    print("   • SVM") 
    print("   • Random Forest")
    print("   • LSTM")
    print("\n🚀 You can now run the Streamlit app with full model comparison!")

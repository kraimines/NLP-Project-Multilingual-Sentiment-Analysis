"""
Script to save trained models from notebook variables
Run this after training your models in the Jupyter notebook
"""

import joblib
import os
import sys

def save_trained_models():
    """
    This function should be called from your Jupyter notebook after training
    """
    try:
        # Create model directory if it doesn't exist
        os.makedirs("model", exist_ok=True)
        
        # Check if models exist in the global namespace (from notebook)
        models_to_save = {}
        
        # Try to get models from globals (when run from notebook)
        if 'pipe_lr' in globals():
            models_to_save['Logistic Regression'] = globals()['pipe_lr']
            print("🔍 Found Logistic Regression model (pipe_lr)")
            
        if 'pipe_svm' in globals():
            models_to_save['SVM'] = globals()['pipe_svm']
            print("🔍 Found SVM model (pipe_svm)")
            
        if 'pipe_rf' in globals():
            models_to_save['Random Forest'] = globals()['pipe_rf']
            print("🔍 Found Random Forest model (pipe_rf)")
            
        if not models_to_save:
            print("❌ No traditional ML models found in global scope.")
            print("💡 Make sure you have trained pipe_lr, pipe_svm, and pipe_rf in your notebook.")
            print("💡 Available variables:", [var for var in globals().keys() if 'pipe' in var or 'model' in var])
            
        # Save each traditional ML model
        file_mapping = {
            'Logistic Regression': 'model/logreg.pkl',
            'SVM': 'model/svm.pkl', 
            'Random Forest': 'model/rf.pkl'
        }
        
        saved_count = 0
        for model_name, model in models_to_save.items():
            try:
                file_path = file_mapping[model_name]
                joblib.dump(model, file_path)
                
                # Calculate accuracy if test data is available
                if 'x_test' in globals() and 'y_test' in globals():
                    accuracy = model.score(globals()['x_test'], globals()['y_test'])
                    print(f"✅ {model_name} saved to {file_path} (Accuracy: {accuracy:.4f})")
                else:
                    print(f"✅ {model_name} saved to {file_path}")
                saved_count += 1
            except Exception as e:
                print(f"❌ Error saving {model_name}: {e}")
        
        # Save LSTM model if available
        lstm_saved = False
        if 'model_lstm' in globals():
            try:
                globals()['model_lstm'].save('model/lstm_model.h5')
                print("✅ LSTM model saved to model/lstm_model.h5")
                lstm_saved = True
                saved_count += 1
            except Exception as e:
                print(f"❌ Error saving LSTM model: {e}")
        else:
            print("⚠️ LSTM model (model_lstm) not found in global scope")
        
        # Save additional components if available
        components_saved = 0
        
        if 'le' in globals():
            try:
                joblib.dump(globals()['le'], 'model/label_encoder.pkl')
                print("✅ Label encoder saved to model/label_encoder.pkl")
                components_saved += 1
            except Exception as e:
                print(f"❌ Error saving label encoder: {e}")
        else:
            print("⚠️ Label encoder (le) not found in global scope")
            
        if 'tokenizer' in globals():
            try:
                joblib.dump(globals()['tokenizer'], 'model/tokenizer.pkl')
                print("✅ Tokenizer saved to model/tokenizer.pkl")
                components_saved += 1
            except Exception as e:
                print(f"❌ Error saving tokenizer: {e}")
        else:
            print("⚠️ Tokenizer not found in global scope")
        
        # Summary
        print(f"\n🎉 Summary:")
        print(f"   📁 {saved_count} models saved")
        print(f"   🔧 {components_saved} components saved")
        
        if saved_count > 0:
            print("✅ Models are now ready for comparison in the Streamlit app!")
            return True
        else:
            print("❌ No models were successfully saved")
            return False
        
    except Exception as e:
        print(f"❌ Error in save_trained_models: {e}")
        return False

def create_sample_models():
    """
    Creates sample models if none exist (for testing purposes)
    """
    print("🔧 Creating sample models for testing...")
    # This would be used if you want to create dummy models for testing
    pass

if __name__ == "__main__":
    print("This script should be run from your Jupyter notebook.")
    print("\n" + "="*60)
    print("📋 COPY THIS CODE TO YOUR JUPYTER NOTEBOOK:")
    print("="*60)
    print("""
# Save all trained models for Streamlit app
import joblib
import os

# Create model directory
os.makedirs("model", exist_ok=True)

# Save traditional ML models (make sure these variables exist in your notebook)
try:
    joblib.dump(pipe_lr, "model/logreg.pkl")
    print("✅ Logistic Regression saved")
except:
    print("❌ pipe_lr not found")

try:
    joblib.dump(pipe_svm, "model/svm.pkl")
    print("✅ SVM saved") 
except:
    print("❌ pipe_svm not found")

try:
    joblib.dump(pipe_rf, "model/rf.pkl")
    print("✅ Random Forest saved")
except:
    print("❌ pipe_rf not found")

# Save LSTM model
try:
    model_lstm.save("model/lstm_model.h5")
    print("✅ LSTM saved")
except:
    print("❌ model_lstm not found")

# Save additional components
try:
    joblib.dump(le, "model/label_encoder.pkl")
    print("✅ Label encoder saved")
except:
    print("❌ le not found")

try:
    joblib.dump(tokenizer, "model/tokenizer.pkl") 
    print("✅ Tokenizer saved")
except:
    print("❌ tokenizer not found")

print("\\n🎉 Model saving complete! Now you can use all models in the Streamlit app!")
    """)
    print("="*60)

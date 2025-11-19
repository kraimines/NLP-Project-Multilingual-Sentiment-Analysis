import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# Page configuration
st.set_page_config(
    page_title="Text Emotion Detector",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load all models
@st.cache_resource
def load_models():
    models = {}
    model_found = False
    
    # Try loading individual models first
    model_files = {
        'Logistic Regression': 'model/logreg.pkl',
        'SVM': 'model/svm.pkl', 
        'Random Forest': 'model/rf.pkl'
    }
    
    for model_name, file_path in model_files.items():
        try:
            models[model_name] = joblib.load(file_path)
            model_found = True
        except FileNotFoundError:
            pass  # Model file doesn't exist, skip it
        except Exception as e:
            st.warning(f"⚠️ Error loading {model_name}: {e}")
    
    # Try loading LSTM if available
    try:
        from tensorflow.keras.models import load_model
        models['LSTM'] = load_model("model/lstm_model.h5")
        models['tokenizer'] = joblib.load("model/tokenizer.pkl") 
        models['label_encoder'] = joblib.load("model/label_encoder.pkl")
        model_found = True
    except FileNotFoundError:
        pass  # LSTM files don't exist
    except Exception as e:
        pass  # LSTM not available, silently continue
    
    # Try loading Transformer if available
    try:
        from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
        models['Transformer'] = DistilBertForSequenceClassification.from_pretrained("model/model/transformer_model")
        models['transformer_tokenizer'] = DistilBertTokenizerFast.from_pretrained("model/model/transformer_tokenizer")
        models['transformer_label_encoder'] = joblib.load("model/model/transformer_label_encoder.pkl")
        model_found = True
    except FileNotFoundError:
        pass  # Transformer files don't exist
    except Exception as e:
        pass  # Transformer not available, silently continue
    
    # Fallback: try loading the original single model
    if not model_found:
        try:
            original_model = joblib.load("model/text_emotion.pkl")
            models['Logistic Regression'] = original_model
            model_found = True
            st.info("ℹ️ Using original trained model (text_emotion.pkl)")
        except Exception as e:
            st.error(f"❌ Error loading fallback model: {e}")
            return None
    
    if not models:
        st.error("❌ No models could be loaded!")
        return None
        
    return models

models = load_models()

# Debug information
if models:
    st.sidebar.success(f"✅ Successfully loaded {len([k for k in models.keys() if k not in ['tokenizer', 'label_encoder']])} model(s)")
else:
    st.sidebar.error("❌ No models loaded")

emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨", "happy": "🤗", 
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔", 
    "shame": "😳", "surprise": "😮"
}

# Emotion color mapping for consistent coloring
emotion_colors = {
    "anger": "#FF6B6B", "disgust": "#8B4513", "fear": "#964B00", 
    "happy": "#FFD93D", "joy": "#FFD93D", "neutral": "#6BCF7F", 
    "sad": "#4D96FF", "sadness": "#4D96FF", "shame": "#FF6B9D", 
    "surprise": "#9B59B6"
}

def predict_emotions(model_name, docx):
    if models is None:
        return None, None
    
    try:
        if model_name == 'LSTM' and 'LSTM' in models:
            # Handle LSTM prediction
            import tensorflow as tf
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            
            model = models['LSTM']
            tokenizer = models['tokenizer']
            le = models['label_encoder']
            
            # Tokenize and pad the text
            sequences = tokenizer.texts_to_sequences([docx])
            padded = pad_sequences(sequences, maxlen=100, padding='post')
            
            # Get prediction
            pred_proba = model.predict(padded)[0]
            pred_class_idx = np.argmax(pred_proba)
            pred_class = le.inverse_transform([pred_class_idx])[0]
            
            return pred_class, pred_proba
        elif model_name == 'Transformer' and 'Transformer' in models:
            # Handle Transformer prediction
            import torch
            
            model = models['Transformer']
            tokenizer = models['transformer_tokenizer']
            le = models['transformer_label_encoder']
            
            # Tokenize the text
            inputs = tokenizer(docx, return_tensors="pt", truncation=True, padding=True, max_length=128)
            
            # Get prediction
            with torch.no_grad():
                outputs = model(**inputs)
                pred_proba = torch.softmax(outputs.logits, dim=-1)[0].numpy()
                pred_class_idx = np.argmax(pred_proba)
                pred_class = le.inverse_transform([pred_class_idx])[0]
            
            return pred_class, pred_proba
        else:
            # Handle traditional ML models
            if model_name in models:
                model = models[model_name]
                results = model.predict([docx])
                
                # Check if model supports predict_proba
                if hasattr(model, 'predict_proba'):
                    try:
                        proba = model.predict_proba([docx])[0]
                        return results[0], proba
                    except Exception as e:
                        st.warning(f"⚠️ {model_name} doesn't support probability predictions: {e}")
                        # Create dummy probabilities for display
                        classes = getattr(model, 'classes_', ['unknown'])
                        dummy_proba = np.zeros(len(classes))
                        pred_index = np.where(classes == results[0])[0]
                        if len(pred_index) > 0:
                            dummy_proba[pred_index[0]] = 1.0
                        return results[0], dummy_proba
                else:
                    st.warning(f"⚠️ {model_name} doesn't support probability predictions")
                    # Create dummy probabilities
                    classes = getattr(model, 'classes_', ['unknown'])
                    dummy_proba = np.zeros(len(classes))
                    pred_index = np.where(classes == results[0])[0]
                    if len(pred_index) > 0:
                        dummy_proba[pred_index[0]] = 1.0
                    return results[0], dummy_proba
            else:
                st.error(f"❌ Model '{model_name}' not found in loaded models")
                return None, None
                
    except Exception as e:
        st.error(f"❌ Prediction error with {model_name}: {e}")
        return None, None

def get_all_model_predictions(docx):
    """Get predictions from all available models"""
    all_predictions = {}
    
    if models is None:
        return all_predictions
    
    for model_name in models.keys():
        if model_name not in ['tokenizer', 'label_encoder']:
            pred, proba = predict_emotions(model_name, docx)
            if pred is not None:
                all_predictions[model_name] = {
                    'prediction': pred,
                    'probability': proba,
                    'confidence': np.max(proba) if proba is not None else 0
                }
    
    return all_predictions

def main():
    # Initialize session state for text input
    if 'input_text' not in st.session_state:
        st.session_state.input_text = ""
    
    # Header with better styling
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .subheader {
            font-size: 1.5rem;
            color: #666;
            text-align: center;
            margin-bottom: 3rem;
        }
        .emotion-card {
            background-color: #f0f2f6;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            border-left: 4px solid #1f77b4;
        }
        .example-btn {
            width: 100%;
            margin: 0.2rem 0;
        }
        .model-comparison {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown('<h1 class="main-header">Text Emotion Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Detect Emotions in Your Text with AI</p>', unsafe_allow_html=True)
    
    # Sidebar for model selection
    st.sidebar.header("🤖 Model Selection")
    
    if models is None:
        st.sidebar.error("❌ No models available")
        return
    
    available_models = [name for name in models.keys() if name not in ['tokenizer', 'label_encoder', 'transformer_tokenizer', 'transformer_label_encoder']]
    
    # Single model selection
    selected_model = st.sidebar.selectbox(
        "Choose a model:",
        options=available_models,
        index=0 if available_models else None
    )
    
    # Model comparison option
    compare_models = st.sidebar.checkbox("🔄 Compare Multiple Models")
    
    if compare_models:
        selected_models = st.sidebar.multiselect(
            "Select models to compare:",
            options=available_models,
            default=available_models[:2] if len(available_models) >= 2 else available_models
        )
    else:
        selected_models = [selected_model] if selected_model else []
    
    # Display model info
    st.sidebar.markdown("### 📊 Model Information")
    model_accuracy = {
        "Logistic Regression": "62.01%",
        "SVM": "62.20%",
        "Random Forest": "56.10%",
        "LSTM": "31.72%",
        "Transformer": "64.90% 🏆"
    }
    
    for model in available_models:
        accuracy = model_accuracy.get(model, "Available")
        st.sidebar.write(f"**{model}**: {accuracy}")
    
    # Show available models
    st.sidebar.markdown("### ✅ Loaded Models")
    st.sidebar.write(f"Found {len(available_models)} model(s)")
      
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter Your Text")
        with st.form(key='emotion_form'):
            # Use session state for the text area value
            raw_text = st.text_area(
                "Type your text here:",
                height=150,
                placeholder="Enter your text to analyze emotions...",
                help="Write a sentence or paragraph to analyze emotional content",
                value=st.session_state.input_text  # This is the key change
            )
            
            col1_1, col1_2, col1_3 = st.columns([1, 1, 1])
            with col1_2:
                submit_text = st.form_submit_button(
                    "Analyze Emotions 🚀",
                    use_container_width=True
                )
    
    with col2:
        st.subheader("💡 Example Texts")
        examples = [
            "I'm so happy today! This is amazing!",
            "I feel really sad about what happened.",
            "That movie was terrifying and scary.",
            "What a surprising turn of events!",
            "This food tastes disgusting."
        ]
        
        for example in examples:
            if st.button(example, key=example, use_container_width=True, 
                        help=f"Click to use: '{example}'"):
                # Update the session state with the example text
                st.session_state.input_text = example
                # Use st.rerun() to refresh the app and show the text in the textarea
                st.rerun()
    
    # If form is submitted and there's text to analyze
    if submit_text and raw_text.strip():
        if not selected_models:
            st.warning("⚠️ Please select at least one model from the sidebar!")
            return
            
        # Show loading state
        with st.spinner("Analyzing emotions with selected models..."):
            if compare_models and len(selected_models) > 1:
                # Multi-model comparison
                all_predictions = get_all_model_predictions(raw_text)
                
                # Results section
                st.markdown("---")
                st.subheader("📊 Model Comparison Results")
                
                # Create columns for each selected model
                cols = st.columns(len(selected_models))
                
                for i, model_name in enumerate(selected_models):
                    with cols[i]:
                        if model_name in all_predictions:
                            pred_data = all_predictions[model_name]
                            prediction = pred_data['prediction']
                            confidence = pred_data['confidence'] * 100
                            
                            # Emotion card for each model
                            st.markdown(f"""
                            <div class="emotion-card">
                                <h4>🤖 {model_name}</h4>
                                <div style="text-align: center; padding: 0.5rem;">
                                    <div style="font-size: 3rem; margin-bottom: 0.5rem;">{emotions_emoji_dict.get(prediction, '🎭')}</div>
                                    <h3 style="color: {emotion_colors.get(prediction, '#1f77b4')}; margin: 0;">{prediction.upper()}</h3>
                                    <p style="font-size: 1rem; color: #666;">Confidence: {confidence:.2f}%</p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error(f"❌ {model_name} prediction failed")
                
                # Comparison table
                st.subheader("📋 Detailed Comparison")
                comparison_data = []
                for model_name in selected_models:
                    if model_name in all_predictions:
                        pred_data = all_predictions[model_name]
                        comparison_data.append({
                            'Model': model_name,
                            'Predicted Emotion': pred_data['prediction'],
                            'Confidence (%)': f"{pred_data['confidence'] * 100:.2f}%",
                            'Emoji': emotions_emoji_dict.get(pred_data['prediction'], '🎭')
                        })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    
                    # Show original text
                    st.info(f"📝 **Original Text:** {raw_text}")
                
            else:
                # Single model prediction
                model_name = selected_models[0]
                prediction, probability = predict_emotions(model_name, raw_text)
                
                if prediction is None:
                    st.error(f"❌ {model_name} prediction failed")
                    return
                
                # Results section
                st.markdown("---")
                st.subheader(f"📊 {model_name} Analysis Results")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    # Emotion card with better styling
                    confidence = np.max(probability) * 100 if probability is not None else 0
                    st.markdown(f"""
                    <div class="emotion-card">
                        <h3>🎭 Predicted Emotion</h3>
                        <div style="text-align: center; padding: 1rem;">
                            <div style="font-size: 4rem; margin-bottom: 1rem;">{emotions_emoji_dict.get(prediction, '🎭')}</div>
                            <h2 style="color: {emotion_colors.get(prediction, '#1f77b4')}; margin: 0;">{prediction.upper()}</h2>
                            <p style="font-size: 1.2rem; color: #666;">Confidence: {confidence:.2f}%</p>
                            <p style="font-size: 1rem; color: #888;">Model: {model_name}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.success("Original Text")
                    st.info(raw_text)
                
                with col4:
                    st.success("Probability Distribution")
                    
                    if probability is not None:
                        # Handle different probability formats
                        if model_name == 'LSTM':
                            # LSTM returns raw probabilities for each class
                            le = models.get('label_encoder')
                            if le:
                                emotions = le.classes_
                                proba_data = list(zip(emotions, probability))
                            else:
                                emotions = ['unknown'] * len(probability)
                                proba_data = list(zip(emotions, probability))
                        else:
                            # Traditional ML models
                            model = models[model_name]
                            emotions = model.classes_
                            proba_data = list(zip(emotions, probability))
                        
                        # Create dataframe
                        proba_df = pd.DataFrame(proba_data, columns=['emotions', 'probability'])
                        proba_df['probability_percent'] = proba_df['probability'] * 100
                        proba_df = proba_df.sort_values('probability', ascending=False)
                        
                        # Create bar chart
                        chart = alt.Chart(proba_df).mark_bar().encode(
                            x=alt.X('probability_percent:Q', title='Probability (%)', axis=alt.Axis(format='.1f')),
                            y=alt.Y('emotions:N', title='Emotions', sort='-x'),
                            color=alt.Color('emotions:N', scale=alt.Scale(
                                domain=list(emotion_colors.keys()),
                                range=list(emotion_colors.values())
                            ), legend=None),
                            tooltip=['emotions:N', 'probability_percent:Q']
                        ).properties(
                            height=400,
                            title=f'{model_name} - Emotion Probability Distribution'
                        )
                        
                        st.altair_chart(chart, use_container_width=True)
                        
                        # Probability table
                        st.success("Detailed Probabilities")
                        display_df = proba_df[['emotions', 'probability_percent']].copy()
                        display_df['probability_percent'] = display_df['probability_percent'].round(2)
                        display_df.columns = ['Emotion', 'Probability (%)']
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    elif submit_text and not raw_text.strip():
        st.warning("⚠️ Please enter some text to analyze!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Built with ❤️ using Streamlit and Machine Learning"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()
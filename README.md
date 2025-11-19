# Text Emotion Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive NLP project that trains, evaluates, and compares multiple machine learning models for detecting emotions from text. The project includes an interactive web application built with Streamlit to showcase the models' predictions.

## Overview

This project provides a full pipeline for a text classification task focused on emotion detection. It demonstrates every step from data cleaning and preprocessing to training various models and deploying them in an interactive app. The primary goal is to compare the performance of classic machine learning algorithms against deep learning and transformer-based architectures.

The models are trained to classify text into one of the following emotions: **joy, sadness, anger, fear, love, and surprise.**

## Features

- **Data Cleaning**: Utilizes the `neattext` library to clean and prepare raw text data.
- **Multiple Model Training**: Trains and evaluates a diverse set of models:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
  - Long Short-Term Memory (LSTM) Network
  - DistilBERT Transformer
- **Model Comparison**: Provides a clear comparison of model accuracy to identify the best-performing architecture.
- **Interactive Web App**: A Streamlit application (`app.py`) allows users to:
  - Input custom text for emotion prediction.
  - Choose a specific model for analysis.
  - View prediction probabilities and confidence scores.

## Project Structure

```
.
├── Text Emotion Detection/
│   ├── app.py                  # Main Streamlit application
│   ├── notebook_save_models.py # Notebook to train and save models
│   ├── data/
│   │   └── emotion_dataset_raw.csv # The raw training data
│   └── model/                  # Saved models (ignored by git)
├── .gitignore              # Specifies files for Git to ignore
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

## Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

- Python 3.8+
- Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kraimines/NLP-Project-Multilingual-Sentiment-Analysis.git
    cd NLP-Project-Multilingual-Sentiment-Analysis
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv .venv
    .\.venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r "Text Emotion Detection/requirements.txt"
    ```

### Model Training

The pre-trained models are not included in this repository due to their large size. You must run the training notebook to generate them.

1.  Open and run the Jupyter Notebook: `Text Emotion Detection/notebook_save_models.py`.
2.  This will train all the models and save the artifacts (`.pkl`, `.h5` files) into the `Text Emotion Detection/model/` directory.

## Usage

Once the models have been trained and saved, you can run the Streamlit application.

1.  Ensure you are in the project's root directory.
2.  Run the following command in your terminal:
    ```bash
    streamlit run "Text Emotion Detection/app.py"
    ```
3.  The application will open in your web browser. You can then enter text and see the emotion predictions from the different models.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

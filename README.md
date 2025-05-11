# 👤 Face Morphing Detection for Identity Verification

This project aims to detect face morphing attempts, which are commonly used for identity verification fraud. It also incorporates facial injury detection to enhance security and analysis of input images, leveraging deep learning techniques for accurate detection.

---

## 📌 Problem Statement

Morphing attacks are a form of identity fraud where two or more facial images are combined to create a new image, misleading identity verification systems. This project focuses on detecting such face morphing attempts and assessing potential facial injuries from images.

---

## 🎯 Goals

- **Face Morphing Detection**: Detect whether the image has been morphed for fraudulent activities.
- **Facial Injury Detection**: Identify facial injuries or signs of accidents to enhance verification.
- **Advanced Image Analysis**: Provide additional metrics on image quality, sharpness, noise, and blurriness to assess the integrity of the input image.

---

## 🛠️ Key Features

- **Face Morphing Detection**: A deep learning model using ResNet50 architecture to distinguish between original and morphed facial images.
- **Facial Injury Detection**: A secondary model that identifies potential facial injuries based on color patterns, edges, and morphology.
- **Image Quality Metrics**: Analyzes image sharpness, blur, and noise levels to help assess the quality of the input image.
- **Performance Metrics**: Tracks inference time and prediction confidence, ensuring the system’s effectiveness and efficiency.

---

## 🔍 Model Architecture

The model uses **ResNet50** as the backbone architecture, pre-trained on ImageNet. It consists of two key parts:

1. **Morphing Detection Model**: Detects if the image is morphed (fraudulent) or original, based on facial features extracted through ResNet50.
2. **Injury Detection Model**: Identifies signs of injury, including cuts or bruises on the face.

### Model Enhancements:
- **Batch Normalization** and **Dropout** layers are used to prevent overfitting and enhance model generalization.
- **Advanced Preprocessing**: Face extraction, color histogram equalization, and noise reduction techniques improve detection accuracy.

---

## 📂 Dataset Description

The model processes images of faces to determine if they have been morphed or altered for fraudulent purposes. The dataset can include both original and morphed images. It works effectively on facial images that are:

- Standard frontal views.
- Affected by injuries or accidents (optional analysis).

---

## 🛠️ Installation and Setup

1. **Uninstall existing versions and install dependencies**:

    ```bash
    !pip uninstall tensorflow keras -y
    !pip install tensorflow==2.12.0 --upgrade
    !pip install keras==2.12.0
    !pip install opencv-python matplotlib seaborn scikit-image gradio --quiet
    ```

2. **Verify installation**: 

    ```python
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {tf.keras.__version__}")
    ```

---

## 📝 Code Overview

### Key Components:

1. **EnhancedFaceMorphingDetector**:
    - A class that initializes the detection models and manages prediction logic.
    - Contains methods for preprocessing images, detecting morphed faces, and detecting injuries.

2. **Image Preprocessing**:
    - Face extraction, resizing to 224x224, and advanced features like histogram equalization and noise reduction are applied to each image before feeding it to the model.

3. **Injury Detection**:
    - Detects facial injuries like cuts or bruises by analyzing color patterns (HSV) and edges using OpenCV.

4. **Prediction**:
    - The `predict_image()` function loads an image, preprocesses it, performs morphing detection, and checks for potential injuries. It also visualizes the analysis and confidence scores.

5. **Performance Metrics**:
    - The system calculates the inference time and displays a **confidence meter** for the prediction, **processing time**, and **injury detection** confidence.

---

## 🔄 Workflow

1. **Model Initialization**:
   - The `EnhancedFaceMorphingDetector` class initializes two models:
     - **Morphing Detection Model**: Detects morphing attacks.
     - **Injury Detection Model**: Identifies facial injuries.
   
2. **Prediction**:
   - Input an image to the system (upload an image file when prompted).
   - The system will preprocess the image, make predictions, and analyze the image for injuries.
   
3. **Results Display**:
   - The system will display the input image, confidence scores, prediction results (Morphed or Original), and potential injury detection.

4. **Visualization**:
   - Confidence distribution, injury detection results (highlighted on the image), and performance metrics will be shown.

---

## 📊 Performance and Metrics

- **Confidence Meter**: Shows a clear confidence score (Original vs. Morphed) using a bar chart.
- **Injury Detection**: If any injuries are detected, they are highlighted, and a confidence score for injury detection is shown.
- **Processing Time**: Measures the time taken to process the image and return the prediction.

---

## 🚀 Running the System

1. **Initialize the system**: Run the code to initialize the `EnhancedFaceMorphingDetector`.
2. **Upload an image**: When prompted, upload a facial image for analysis.
3. **View Results**: The system will display:
   - Morphing detection result: "Morphed" or "Original".
   - Injury detection: Warning if injuries are detected.
   - Performance metrics: Confidence, processing time, etc.

---

## 👨‍💻 Developed By

**K Keerthi**  
Computer Science Engineer | AI & ML Enthusiast

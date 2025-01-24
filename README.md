# ML-Based Medicine Identification & Disease Prediction

## Project Overview
This project aims to develop a machine learning-based system that can:
1. **Identify the name and use of a medicine** when shown to a camera.
2. **Predict diseases** based on user-input symptoms, weather conditions, and location.

The project leverages **computer vision, NLP, and predictive modeling** techniques to assist users in real-time medical guidance.

### Team Members
- **22051223**
- **22051664**
- **22053921**
- **2205907**
- **22051232**
- **22051310**

---
## **Part 1: Medicine Identification using Computer Vision**
### **Goal**
Develop a model that identifies medicines and their uses when shown to a camera.

### **Tasks**
#### **Dataset Collection**
- Collect images of different medicines, including labels and packaging.
- Gather metadata (medicine name, composition, uses, etc.).
- Apply data augmentation techniques to improve robustness.

#### **Preprocessing & Feature Extraction**
- Image resizing, normalization, and enhancement.
- Use **Optical Character Recognition (OCR)** for extracting text from labels.

#### **Model Selection & Training**
- Utilize pre-trained CNN models such as **ResNet, VGG16, EfficientNet** for image classification.
- Fine-tune the model using a curated medicine dataset.

#### **Deployment**
- Develop an **API** to integrate with a mobile or web app for real-time predictions.

### **Technologies & Libraries**
- **Languages**: Python
- **Libraries**: OpenCV, TensorFlow/Keras, PyTorch, Tesseract OCR, PIL, NumPy, Pandas
- **Tools**: Jupyter Notebook, Google Colab

---
## **Part 2: Disease Prediction using Symptoms, Weather & Location**
### **Goal**
Predict possible diseases based on user-input symptoms, current weather conditions, and location.

### **Tasks**
#### **Data Collection**
- Use datasets containing **symptoms and corresponding diseases** (Kaggle, WHO, CDC).
- Collect weather data using **OpenWeatherMap API**.
- Gather location-based health reports (local disease outbreaks, flu trends).

#### **Feature Engineering & Preprocessing**
- Encode categorical symptoms.
- Normalize weather and geographical data.
- Handle missing values using imputation techniques.

#### **Model Training & Evaluation**
- Train models using **Decision Trees, Random Forest, XGBoost, and Neural Networks**.
- Implement **NLP-based symptom analysis** using Transformers/BERT.
- Evaluate models using **precision-recall, accuracy, MSE, RMSE, and MAE**.

#### **API Development & Deployment**
- Develop a **REST API** to process user inputs and return disease predictions.

### **Technologies & Libraries**
- **Languages**: Python
- **Libraries**: Scikit-learn, Pandas, NumPy, TensorFlow/Keras, XGBoost, Flask/FastAPI, Transformers (Hugging Face)
- **APIs**: OpenWeatherMap API, Geolocation APIs

---
## **Part 3: Integration & Application Development**
### **Goal**
Build a **user-friendly interface** where users can upload medicine images or input symptoms to get predictions.

### **Tasks**
#### **Front-End Development**
- Develop a **web or mobile app interface**.
- Implement a **form for symptom entry** and real-time camera access for medicine identification.

#### **Back-End Development & API Integration**
- Connect the **ML models** with the front end via REST APIs.
- Use **Flask, Django, or FastAPI** for back-end logic.

#### **Testing & Deployment**
- Test the system with real-world data.
- Deploy using **AWS/GCP/Azure** or Firebase.

### **Technologies & Libraries**
- **Front-end**: React.js, Flutter, or Android (Kotlin)
- **Back-end**: Flask, FastAPI, or Django
- **Database**: PostgreSQL, Firebase, MongoDB
- **Cloud Deployment**: AWS, GCP, or Azure

---
## **Notes**
- Each directory in `./src` consists of one of the problems assigned.
- Python version management is flexible, but we use **[pyenv](https://github.com/pyenv/pyenv)** as the default.
- Use **[venv](https://docs.python.org/3/library/venv.html)** for requirement management.
- Users should install the required Python version (found in `.python-version`).
- After setting up a virtual environment, install dependencies and proceed to the `src/` directory to evaluate the project.

---
## **Conclusion**
This project integrates **computer vision, NLP, and predictive modeling** to assist in medicine recognition and disease prediction. By leveraging real-time data and deep learning models, we aim to provide an accessible and effective solution for users seeking preliminary medical guidance.


# 😃 Facial Emotion Recognition using CNN

A deep learning-based facial emotion classifier that can detect **7 emotions** from facial images. This project is built using **Keras**, **TensorFlow**, **OpenCV**, and is trained on the **FER2013** dataset.

---

## 🚀 Features

- 🧠 Trained CNN with 4 convolutional layers
- 🧪 Tested on FER2013 test set
- 🎯 Recognizes 7 emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- 🎥 Real-time webcam emotion prediction
- 💾 Pre-trained model weights available

---

## 🛠️ Tech Stack

- **Language**: Python  
- **Libraries**: TensorFlow, Keras, NumPy, Pandas, OpenCV, tqdm  
- **Tools**: Jupyter Notebook, OpenCV Webcam, Google Drive (for model storage)

---

---

## 🧠 Model Overview

- ✅ **4 Convolutional Layers** with ReLU activation  
- ✅ **MaxPooling + Dropout** after each conv layer  
- ✅ **2 Dense (Fully Connected) Layers**  
- ✅ **Output Layer** with 7 neurons and Softmax activation  
- ✅ Trained with Categorical Crossentropy & Adam Optimizer  

---

## 🔗 Resources

- 📦 Dataset: [FER2013 on Kaggle](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)  
- 📥 Trained Model Files: [Google Drive](https://drive.google.com/drive/folders/1-SCxdgtrnEaK2iX6_YX0mIO0siRGS2WN?usp=drive_link)

---

## 💻 Setup & Installation

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/emotion-recognition.git
cd emotion-recognition
```
## 💻 Setup & Installation

### 2. Create a Virtual Environment (Optional)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3. Install Requirements
Install all required Python libraries using the command below:

```bash
pip install -r requirements.txt
```

### 4. Download Trained Model
Download the pre-trained model files from [Google Drive](https://drive.google.com/drive/folders/1-SCxdgtrnEaK2iX6_YX0mIO0siRGS2WN?usp=drive_link) and place them in the root directory of your project:
- `facialemotionmodel.json` — Model architecture
- `facialemotionmodel.h5` — Model weights

### 5. Run the Notebook
To train or test the model, open the Jupyter notebook:

```bash
jupyter notebook modeltrain.ipynb



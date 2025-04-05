# Traffic-Sign-Classification
Traffic Sign Classification Using Tensor Flow and Keras in CNN 
Here's a clean and professional `README.md` file for your GitHub repository based on the documentation you provided for the **Traffic Sign Classification** project:lllll

---

markdown
# 🚦 Traffic Sign Classification using CNN

This project demonstrates how to build a **Traffic Sign Classification** system using **Convolutional Neural Networks (CNNs)** with **TensorFlow** and **Keras**. It’s a multi-class classification problem, designed to help machines recognize road signs — a crucial component in autonomous driving systems.

---

## 📁 Dataset

We used a traffic sign dataset containing **43 classes** with training and testing sets.

Steps:

- Downloaded and unzipped the dataset.
- Cleaned the dataset by removing unnecessary files.
- Visualized random samples to understand the data.
- Measured and analyzed image dimensions to check consistency.
- Resized all images to **50x50 pixels** for uniformity.

---

## 🧪 Exploratory Data Analysis

- Displayed random test images in a 4x4 grid with height and width labels.
- Checked class distribution to confirm data is **balanced**.
- Plotted histograms of image height and width to analyze variation.
- Normalized image pixel values (0–255 → 0–1).

---

## 🧠 Model Architecture (CNN)

The CNN is built using `Sequential()` API in Keras. Here's the architecture:

1. **Conv2D** + **MaxPooling** + **Dropout**
2. **Conv2D** + **MaxPooling** + **Dropout**
3. **Flatten**
4. **Dense(128, ReLU)** + **Dropout**
5. **Dense(43, Softmax)** – for 43 traffic sign classes

All layers use ReLU activation, except the final one which uses Softmax.

---

## 📊 Training & Evaluation

- Used one-hot encoding for the labels via `to_categorical()`.
- Model trained using the preprocessed dataset.
- Plotted training accuracy to monitor model performance across epochs.

---

## 🔧 Libraries & Tools Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib / Seaborn
- Pandas
- PIL (Python Imaging Library)
- Google Colab (for training)

---

## 📌 Highlights

- Dataset contains **39209 training images**.
- Images resized and normalized for consistency.
- Model uses **Dropout** layers to prevent overfitting.
- Accuracy tracked using training history.

---

## 📂 Folder Structure


traffic_sign_dataset/
│

├── Train/

│   ├── 0/

│   ├── 1/

│   └── ...  (up to 42)

│
└── Test/

    ├── test_image_1.png
    
    ├── 

    




## ✅ Output

- Final model capable of classifying unseen traffic sign images.
- 4x4 image grid showing random predictions with dimensions.
- Accuracy plots show model learning over time.



## 🛠 Future Improvements

- Add confusion matrix for evaluation.
- Integrate with a real-time traffic sign detection pipeline.
- Convert model to TensorFlow Lite for deployment on edge devices.



## 📬 Contact

For feedback or collaboration, feel free to open an issue or contact the project owner.
Email: noorullahzamindar007@gmail.com
W-No: +93797529779



**Happy Coding! 🚀**


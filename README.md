# 🧵 Pattern Sense

**Pattern Sense** is a deep learning web application that classifies fabric patterns into categories like **floral**, **striped**, and **plain**. It is built using **Flask** and **TensorFlow**, providing a simple and elegant user interface for uploading and predicting fabric patterns.

---

## 🚀 Features

- Upload fabric images through a web UI
- Automatically classifies into one of the trained fabric pattern categories
- Responsive interface with a textile-themed background
- Result page displays uploaded image and predicted pattern

---

## 🖥️ Technologies Used

- Python
- Flask (Backend Web Framework)
- TensorFlow + Keras (CNN Model)
- HTML, CSS (Frontend)
- Bootstrap (for result page styling)

---

## 📁 Project Structure

pattern_sense/
├── app.py
├── train_model.py
├── model/
│ └── model_cnn.h5
├── static/
│ ├── styles.css
│ ├── images/
│ │ └── background.jpg
│ └── uploads/
├── templates/
│ ├── home.html
│ ├── upload.html
│ └── result.html
├── screenshots/
│ ├── home.png
│ ├── upload.png
│ └── result.png
├── requirements.txt
└── README.md


---
## 📸 Screenshots

### 🔹 Home Page
![Home Page](screenshots/home_page.png)

### 🔹 Upload Page
![Upload Page](screenshots/upload_page.png)

### 🔹 Result Page
![Result Page](screenshots/result_page.png)



---

## 🧪 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/rathan444/pattern-sense.git
cd pattern-sense


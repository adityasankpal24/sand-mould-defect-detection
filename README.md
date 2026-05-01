# Sand Mould Defect Detection System 🏭🔍

## 🚀 Overview

This project implements a **real-time industrial defect detection system** for sand moulds using computer vision techniques.
The system detects surface defects such as **cracks, dents, missing sand, and irregularities** using image processing.

---

## 🎯 Key Features

* 🔍 Real-time defect detection using CCTV/IP cameras
* 🧠 Frame differencing with reference mould comparison
* ⚙️ Image processing pipeline (OpenCV)
* 📦 Contour-based defect localization
* ⚡ Real-time performance (~20–30 FPS)
* 💰 Cost-effective solution (no expensive sensors required)

---

## 🛠 Tech Stack

* Python
* OpenCV
* NumPy
* Image Processing Techniques

---

## ⚙️ Working Principle

1. Capture live video using IP cameras
2. Create a reference (defect-free) mould image
3. Convert frames to grayscale
4. Apply Gaussian blur for noise reduction
5. Perform frame differencing
6. Apply thresholding for segmentation
7. Detect contours to identify defects
8. Highlight defects using bounding boxes

---

## 📷 Project Demo

### 🧠 System Design

(Add your flowchart here)

### 💻 Implementation

(Add code/output screenshot here)

### 🚀 Output

(Add defect detection image here)

---

## 📊 Performance

* Real-time processing: ~20–30 FPS
* Reliable detection for visible surface defects
* Works best under controlled lighting conditions

---

## 🔮 Future Improvements

* Integration with deep learning models (CNN)
* Adaptive thresholding for varying lighting
* Embedded deployment (Raspberry Pi / Jetson Nano)
* Automated rejection system in conveyor setup

---

## 📌 Applications

* Foundry industry
* Quality inspection systems
* Industrial automation

---

## 👨‍💻 Author

Aditya Sankpal

---

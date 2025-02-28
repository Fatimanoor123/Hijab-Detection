# Hijab Detection using YOLOv8

This project performs real-time **Hijab Detection** using **YOLOv8** on a video file. The model detects whether a person is wearing a hijab and draws bounding boxes with confidence scores.

## 📌 Requirements
Ensure you have the following dependencies installed:

```bash
pip install ultralytics opencv-python numpy
```

## 📂 Model and Video Setup
- Place your trained **YOLOv8 model** (`best.pt`) in the same directory as the script.
- Provide the path to your **video file** (`video.mp4`).

## 🚀 Usage
Run the script using:

```bash
python hijab_detection.py
```

## 🎯 Features
- Detects whether a person is **wearing a hijab or not**
- Works on **video input**
- Displays **bounding boxes with confidence scores**
- Resizes frames while maintaining **aspect ratio**

## 🛠 Troubleshooting
- If detection is not accurate, try **retraining the model** with a balanced dataset.
- Ensure **best.pt** is in the same directory as the script.
- If the video does not play, check the **video path** and format.

## 📜 License
This project is open-source. Feel free to modify and improve!

## Output
![image](https://github.com/user-attachments/assets/8574a128-bbd7-422f-aa3e-b3575af4997f)

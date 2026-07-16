# Vision-Based-TicTacToe
# 🎮 Air Tic-Tac-Toe using Computer Vision

A real-time **gesture-controlled Tic-Tac-Toe game** built with **Python, OpenCV, and MediaPipe**. The game allows **two players** to play on the same webcam without using a mouse or keyboard. Players simply point with their index finger to select a cell on the board.

---

## 📌 Features

- ✋ Real-time hand tracking using **MediaPipe Tasks API**
- 👥 Two-player mode using a single webcam
- 🎯 Finger gesture interaction
- ⏳ Hold-to-select mechanism to avoid accidental clicks
- 🏆 Automatic winner detection
- 📊 Live score tracking
- 🎨 Modern graphical interface built with OpenCV
- 🔄 Restart game without restarting the application

---

## 🖥️ Demo

### Gameplay

- Left Hand → **Player X**
- Right Hand → **Player O**
- Point your index finger to an empty cell.
- Hold your finger for **0.5 seconds** to place your mark.
- First player to align three symbols wins.

---

## 🛠️ Technologies Used

- Python 3.12
- OpenCV
- MediaPipe Tasks API
- NumPy

---

## 📂 Project Structure

```
Air-TicTacToe/
│
├── main.py
├── hand_landmarker.task
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/Air-TicTacToe.git

cd Air-TicTacToe
```

---

### Create Virtual Environment

```bash
python -m venv venv
```

Activate it

Windows

```bash
venv\Scripts\activate
```

Linux/Mac

```bash
source venv/bin/activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

or

```bash
pip install mediapipe opencv-python numpy
```

---

## ▶️ Run

```bash
python main.py
```

The application will automatically download the MediaPipe hand landmark model the first time it runs.

---

## 🎮 Controls

| Key | Function |
|------|----------|
| **R** | Restart current game |
| **Q** | Quit application |

---

## 🧠 How It Works

1. Webcam captures live video.
2. MediaPipe detects up to **two hands** simultaneously.
3. The detected hand is classified as **Left** or **Right**.
4. Left Hand is assigned to **Player X**.
5. Right Hand is assigned to **Player O**.
6. The index fingertip position is tracked continuously.
7. When a fingertip remains inside an empty cell for **0.5 seconds**, the move is confirmed.
8. The game checks for:
   - Winner
   - Draw
   - Score update
9. The board updates in real time.

---

## 🏗️ System Architecture

```
Webcam
   │
   ▼
OpenCV Video Capture
   │
   ▼
MediaPipe Hand Detection
   │
   ▼
Index Finger Tracking
   │
   ▼
Cell Selection
   │
   ▼
Game Logic
   │
   ├── Player Turn
   ├── Winner Detection
   ├── Draw Detection
   └── Score Update
   │
   ▼
OpenCV GUI Rendering
```

---

## 📸 Screenshots

Add screenshots here.

Example:

```
images/
├── gameplay.png
├── winner.png
└── menu.png
```

---

## 📋 Requirements

- Python 3.12
- Webcam
- OpenCV
- MediaPipe
- NumPy

---



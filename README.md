# Air Tic Tac Toe ✋⭕❌

A two-player Tic Tac Toe game controlled entirely with **hand gestures**, using a single webcam and [MediaPipe](https://developers.google.com/mediapipe) hand tracking. No mouse, no keyboard, no controllers — just point your index finger at a cell and hold it.

![Python](https://img.shields.io/badge/python-3.9%20--%203.12-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-required-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks%20API-orange)

---

## ✨ Features

- **Two players, one camera** — both hands are tracked simultaneously.
  - Left hand → **Player 1 (X)**
  - Right hand → **Player 2 (O)**
- **Live "Hand Detected" status badges** for each player, with a pulsing indicator dot, so you always know whether the camera can see you.
- **Lobby + 3‑2‑1 countdown** — the round only starts once both players' hands are detected.
- **Best-of-3 match play** with round score, match score, and a "Match Point" callout.
- **Smoothed cursor tracking** (jitter-free fingertip pointer).
- **Win celebration** particle effects.
- **Live FPS counter** and an in-game keyboard shortcuts overlay (`h`).
- **Auto-fit window** — automatically sizes itself to your screen resolution and stays fully on-screen; the window is also resizable.
- Clean, modern "ICA Academy" themed UI built entirely with OpenCV drawing primitives — no external UI framework needed.

---

## 🎮 How to Play

1. Sit (or stand) so both hands can be seen by the webcam.
2. Wait for **both** player badges to turn green — that means both hands are detected.
3. A 3‑2‑1 countdown starts automatically once both players are ready.
4. Point your index finger at an empty cell and **hold it there for ~0.5 seconds** to place your mark.
5. Play alternates automatically between Player 1 (X) and Player 2 (O).
6. First to win **3 rounds** wins the match.

### Controls

| Key | Action |
|-----|--------|
| `r` | Restart the current round (keeps match score) |
| `m` | Start a brand new match (resets match score) |
| `h` | Show / hide the keyboard shortcuts panel |
| `q` | Quit |

---

## ⚙️ Requirements

- Python **3.9 – 3.12** (MediaPipe does not yet ship official wheels for Python 3.13+)
- A working webcam
- Packages:
  - `mediapipe`
  - `opencv-python`
  - `numpy`

> `tkinter` is used internally to auto-detect your screen resolution. It ships with the standard Python installer on Windows/macOS; on Linux you may need to install it separately (e.g. `sudo apt install python3-tk`). If it's unavailable, the game falls back to a safe default window size.

---

## 🚀 Installation & Setup

### 1. Create a Python 3.12 virtual environment

**Windows**
```bash
py -3.12 -m venv venv312
venv312\Scripts\activate
```

**macOS / Linux**
```bash
python3.12 -m venv venv312
source venv312/bin/activate
```

### 2. Install dependencies

```bash
pip install mediapipe opencv-python numpy
```

### 3. Run the game

```bash
python air_tic_tac_toe.py
```

The hand-tracking model (`hand_landmarker.task`) is downloaded automatically on first run and cached locally — no manual setup needed.

---

## 🖼️ Project Structure

```
.
├── air_tic_tac_toe.py     # Main game script
├── hand_landmarker.task   # Auto-downloaded on first run (not committed)
└── README.md
```

---

## 🧠 How It Works

- **Hand tracking**: MediaPipe's `HandLandmarker` (Tasks API) runs in `VIDEO` mode with `num_hands=2`, detecting up to two hands per frame along with a Left/Right handedness label.
- **Player assignment**: The handedness label permanently maps *Left → Player 1 (X)* and *Right → Player 2 (O)*, so it doesn't matter which hand appears first or briefly leaves the frame.
- **Selection**: The index fingertip landmark (landmark #8) is tracked and smoothed (exponential moving average) to reduce jitter. Hovering over an empty cell for `threshold_time` (0.5s) places that player's mark.
- **Rendering**: The entire UI (header, badges, board, marks, particles) is drawn each frame directly onto the camera feed using OpenCV primitives, with the ICA Academy color palette (BGR).

---

## 🛠️ Troubleshooting

| Issue | Fix |
|---|---|
| `ERROR: Could not open a webcam` | Make sure no other app is using the camera and that it's connected. |
| Window doesn't fit my screen | The window now auto-fits your screen resolution on launch and is resizable — drag the edges if you need a different size. |
| MediaPipe install fails | Confirm you're using Python 3.9–3.12, not 3.13+. |
| Hand not detected | Improve lighting and keep your hand fully inside the camera frame; the detection badge will turn green once tracking locks on. |

---

## 🙌 Credits

Built with [MediaPipe](https://developers.google.com/mediapipe) and [OpenCV](https://opencv.org/) .

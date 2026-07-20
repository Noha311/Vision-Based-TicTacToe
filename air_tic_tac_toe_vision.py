"""
ICA Academy - Virtual Hand-Tracking Tic Tac Toe (Two Players, One Camera)
==========================================================================
Compatible with the modern MediaPipe Tasks API (mediapipe>=0.10.x).

IMPORTANT (compatibility note):
MediaPipe currently only ships official wheels for Python 3.9 - 3.12.
There is NO official MediaPipe wheel for Python 3.13 or 3.14 yet.
To run this script you must use a Python 3.12 virtual environment,
even if your system's default Python is 3.14.

Setup (Windows example):
    py -3.12 -m venv venv312
    venv312\\Scripts\\activate
    pip install mediapipe opencv-python numpy

Then run this script using that same venv's python.exe.

HOW TWO-PLAYER MODE WORKS
--------------------------
- The hand landmarker runs with num_hands=2, so BOTH players' hands are
  tracked at the same time from a single webcam.
- MediaPipe reports a "handedness" label (Left / Right) for every hand
  it sees. That label is used to permanently assign:
      Left hand  -> Player 1 / X (orange accent)
      Right hand -> Player 2 / O (yellow accent)
  so it doesn't matter which hand shows up first in a given frame, or
  if a hand briefly leaves the frame and comes back.
- The game is still turn-based (classic Tic Tac Toe rules), but since
  both hands are tracked live, players don't have to take turns handing
  over a mouse/keyboard - whoever's turn it is just points at a cell
  and holds their finger there to place their mark.
- Point your index finger at an empty cell and hold it for half a
  second to place your mark.

WHAT'S NEW IN THIS VERSION
---------------------------
1. Live "Player Detected" status badges for both players, with a
   pulsing green/red indicator dot, so each person can immediately see
   whether the camera has locked on to their hand.
2. A "Waiting for players" lobby screen with animated dots that only
   releases into a 3-2-1 countdown once BOTH hands have been detected -
   no more starting a round with a hand the camera can't see.
3. Best-of-N match play (first to MATCH_TARGET round wins takes the
   match), with a live match score panel and a "Match Point" callout.
4. Jitter-reduced cursors: fingertip positions are smoothed with an
   exponential moving average so the pointer doesn't shake.
5. A short particle-burst celebration animation plays over the winning
   line when a round is won.
6. Live FPS counter and an optional full key-map help overlay (toggle
   with 'h').
7. Friendlier startup/shutdown behavior: a clear error message (instead
   of a silent crash) if no webcam is found, and an end-of-session
   summary printed to the console when you quit.

CONTROLS
--------
  r  -> restart the current round (keeps match score)
  m  -> start a brand new match (resets match score)
  h  -> show / hide the full key-map help panel
  q  -> quit
"""

import os
import sys
import time
import random
import urllib.request

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# ---------------------------------------------------------------------
# 1) Download the hand landmarker model automatically if missing
# ---------------------------------------------------------------------
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task model, please wait...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

# ---------------------------------------------------------------------
# 2) Create the HandLandmarker (new Tasks API) in VIDEO mode, TWO hands
# ---------------------------------------------------------------------
BaseOptions = mp_tasks.BaseOptions
HandLandmarker = mp_vision.HandLandmarker
HandLandmarkerOptions = mp_vision.HandLandmarkerOptions
VisionRunningMode = mp_vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open a webcam (index 0). "
          "Check that a camera is connected and not in use by another app.")
    landmarker.close()
    sys.exit(1)

# ---------------------------------------------------------------------
# 3) ICA Academy color palette
#    NOTE: OpenCV uses BGR order, so tuples below are (B, G, R).
# ---------------------------------------------------------------------
BG_DARK = (26, 16, 8)
CARD_BG = (46, 26, 18)
CARD_BG_HOVER = (72, 40, 26)
CARD_BORDER = (60, 45, 30)
TEXT_WHITE = (245, 246, 248)
TEXT_MUTED = (167, 147, 139)
TEXT_DARK = (18, 16, 14)

ACCENT_X = (255, 108, 59)    # Player 1 / X - "orange" end of the ICA gradient
ACCENT_O = (238, 211, 34)    # Player 2 / O - "yellow/cyan" end of the ICA gradient
ACCENT_WIN = (90, 214, 120)  # highlight color for the winning line
ACCENT_BAD = (86, 86, 220)   # "not detected" red

CORNER_RADIUS = 16
threshold_time = 0.5

# ---------------------------------------------------------------------
# 3b) Auto-fit the window to the user's screen
#    (fixes the "only the first two rows are visible" issue on laptops
#    with a smaller screen than 1440x900 - the whole board now always
#    fits on screen, and the window can still be resized manually).
# ---------------------------------------------------------------------
_BASE_W, _BASE_H = 1440, 900
try:
    import tkinter as _tk
    _root = _tk.Tk()
    _root.withdraw()
    SCREEN_W = _root.winfo_screenwidth()
    SCREEN_H = _root.winfo_screenheight()
    _root.destroy()
except Exception:
    SCREEN_W, SCREEN_H = 1440, 900

_scale = min((SCREEN_W * 0.92) / _BASE_W, (SCREEN_H * 0.85) / _BASE_H, 1.0)
FRAME_W, FRAME_H = int(_BASE_W * _scale), int(_BASE_H * _scale)

LABEL_TO_PLAYER = {"Left": "X", "Right": "O"}
PLAYER_ACCENT = {"X": ACCENT_X, "O": ACCENT_O}
PLAYER_DISPLAY_NAME = {"X": "PLAYER 1 - X", "O": "PLAYER 2 - O"}
MATCH_TARGET = 3           # first to win this many rounds wins the match
CURSOR_SMOOTHING = 0.35    # EMA smoothing factor for fingertip position (0-1, higher = snappier)
COUNTDOWN_SECONDS = 3
CELEBRATION_SECONDS = 1.6


def lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


# ---------------------------------------------------------------------
# 4) Board layout - 3x3 grid, centered under the header/status panel
# ---------------------------------------------------------------------
CELL_SIZE = max(90, int(170 * _scale))
CELL_GAP = max(8, int(16 * _scale))
BOARD_SIZE = CELL_SIZE * 3 + CELL_GAP * 2
BOARD_X = (FRAME_W - BOARD_SIZE) // 2
BOARD_Y = int(300 * _scale)

cells = []
for r in range(3):
    row = []
    for c in range(3):
        x1 = BOARD_X + c * (CELL_SIZE + CELL_GAP)
        y1 = BOARD_Y + r * (CELL_SIZE + CELL_GAP)
        row.append({
            "row": r, "col": c,
            "start": (x1, y1),
            "end": (x1 + CELL_SIZE, y1 + CELL_SIZE),
        })
    cells.append(row)

WIN_LINES = [
    [(0, 0), (0, 1), (0, 2)], [(1, 0), (1, 1), (1, 2)], [(2, 0), (2, 1), (2, 2)],
    [(0, 0), (1, 0), (2, 0)], [(0, 1), (1, 1), (2, 1)], [(0, 2), (1, 2), (2, 2)],
    [(0, 0), (1, 1), (2, 2)], [(0, 2), (1, 1), (2, 0)],
]


# ---------------------------------------------------------------------
# 5) Game state
# ---------------------------------------------------------------------
def new_board():
    return [[None] * 3 for _ in range(3)]


board = new_board()
current_player = "X"

# game_state machine: "waiting" -> "countdown" -> "playing" -> "won"/"draw"
game_state = "waiting"
winner = None
winning_line = None
scores = {"X": 0, "O": 0}
rounds_played = 0
match_complete = False
match_winner = None

hover_cell = {"X": None, "O": None}
hover_start = {"X": None, "O": None}

detected = {"X": False, "O": False}
smoothed_pos = {"X": None, "O": None}

countdown_start_time = None
celebration_start_time = None
particles = []

show_help = False

fps = 0.0
_prev_frame_time = time.time()


def check_winner(b):
    for line in WIN_LINES:
        vals = [b[r][c] for (r, c) in line]
        if vals[0] is not None and vals[0] == vals[1] == vals[2]:
            return vals[0], line
    return None, None


def is_full(b):
    return all(b[r][c] is not None for r in range(3) for c in range(3))


def spawn_particles(center, color, count=48):
    for _ in range(count):
        angle = random.uniform(0, 2 * np.pi)
        speed = random.uniform(120, 420)
        particles.append({
            "pos": [float(center[0]), float(center[1])],
            "vel": [np.cos(angle) * speed, np.sin(angle) * speed],
            "color": color,
            "life": random.uniform(0.8, CELEBRATION_SECONDS),
            "max_life": CELEBRATION_SECONDS,
            "radius": random.uniform(3, 6),
        })


def update_particles(dt):
    for p in particles:
        p["pos"][0] += p["vel"][0] * dt
        p["pos"][1] += p["vel"][1] * dt
        p["vel"][1] += 320 * dt  # gravity
        p["life"] -= dt
    particles[:] = [p for p in particles if p["life"] > 0]


def draw_particles(img):
    for p in particles:
        alpha = max(0.0, min(1.0, p["life"] / p["max_life"]))
        color = lerp_color(BG_DARK, p["color"], alpha)
        cv2.circle(img, (int(p["pos"][0]), int(p["pos"][1])), int(p["radius"]), color, -1, cv2.LINE_AA)


def reset_round(keep_scores=True):
    global board, current_player, game_state, winner, winning_line
    global hover_cell, hover_start, particles, celebration_start_time
    board = new_board()
    current_player = "X"
    game_state = "playing" if (detected["X"] and detected["O"]) else "waiting"
    winner = None
    winning_line = None
    hover_cell = {"X": None, "O": None}
    hover_start = {"X": None, "O": None}
    particles = []
    celebration_start_time = None
    if not keep_scores:
        scores["X"] = 0
        scores["O"] = 0


def start_new_match():
    global rounds_played, match_complete, match_winner
    rounds_played = 0
    match_complete = False
    match_winner = None
    reset_round(keep_scores=False)


# ---------------------------------------------------------------------
# 6) Drawing helpers
# ---------------------------------------------------------------------
def draw_rounded_rect(img, top_left, bottom_right, color, radius=CORNER_RADIUS, thickness=-1):
    x1, y1 = top_left
    x2, y2 = bottom_right
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)


def draw_gradient_strip(img, x1, x2, y1, y2, c1, c2, segments=60):
    step = max(1, (x2 - x1) // segments)
    for sx in range(x1, x2, step):
        t = (sx - x1) / max(1, (x2 - x1))
        cv2.rectangle(img, (sx, y1), (min(sx + step, x2), y2), lerp_color(c1, c2, t), -1)


def draw_pill_badge(img, center_x, y, text, color):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    pad_x, pad_y = 18, 10
    w = text_size[0] + pad_x * 2
    h = text_size[1] + pad_y * 2
    x1 = center_x - w // 2
    draw_rounded_rect(img, (x1, y), (x1 + w, y + h), CARD_BG, radius=h // 2)
    draw_rounded_rect(img, (x1, y), (x1 + w, y + h), color, radius=h // 2, thickness=1)
    cv2.circle(img, (x1 + 16, y + h // 2), 4, color, -1)
    cv2.putText(img, text, (x1 + pad_x + 10, y + h - pad_y + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_WHITE, 1, cv2.LINE_AA)


def draw_header(img):
    draw_gradient_strip(img, 90, FRAME_W - 90, 18, 24, ACCENT_X, ACCENT_O)

    cv2.putText(img, "</> ICA ACADEMY", (90, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, TEXT_WHITE, 2, cv2.LINE_AA)
    draw_pill_badge(img, FRAME_W // 2, 46, "TWO PLAYERS - LIVE", ACCENT_O)

    fps_text = f"{fps:4.1f} FPS"
    text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.putText(img, fps_text, (FRAME_W - 90 - text_size[0], 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_MUTED, 1, cv2.LINE_AA)

    cv2.putText(img, "Air Tic Tac Toe", (90, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, TEXT_WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, "Left hand = Player 1 (X)   |   Right hand = Player 2 (O)   |   hold finger on a cell to play",
                (90, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_MUTED, 1, cv2.LINE_AA)


def draw_detection_panel(img):
    """Two side-by-side cards, one per player, showing live detection status."""
    panel_y1, panel_y2 = int(148 * _scale), int(210 * _scale)
    card_w = (FRAME_W - 180 - CELL_GAP) // 2
    pulse = 0.5 + 0.5 * np.sin(time.time() * 6.0)

    for i, player in enumerate(["X", "O"]):
        x1 = 90 + i * (card_w + CELL_GAP)
        x2 = x1 + card_w
        is_on = detected[player]
        accent = PLAYER_ACCENT[player]
        status_color = accent if is_on else ACCENT_BAD

        draw_rounded_rect(img, (x1, panel_y1), (x2, panel_y2), CARD_BG, radius=14)
        border_c = lerp_color(CARD_BORDER, status_color, 0.6 if is_on else 0.25)
        draw_rounded_rect(img, (x1, panel_y1), (x2, panel_y2), border_c, radius=14, thickness=2)

        # pulsing status dot
        dot_radius = 7 if not is_on else int(6 + 3 * pulse)
        cv2.circle(img, (x1 + 24, panel_y1 + 31), dot_radius, status_color, -1, cv2.LINE_AA)
        if is_on:
            cv2.circle(img, (x1 + 24, panel_y1 + 31), dot_radius + 5, status_color, 1, cv2.LINE_AA)

        cv2.putText(img, PLAYER_DISPLAY_NAME[player], (x1 + 42, panel_y1 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_WHITE, 1, cv2.LINE_AA)
        status_text = "HAND DETECTED" if is_on else "NOT DETECTED"
        cv2.putText(img, status_text, (x1 + 42, panel_y1 + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1, cv2.LINE_AA)


def draw_status_panel(img):
    x1, y1, x2, y2 = 90, int(222 * _scale), FRAME_W - 90, int(280 * _scale)

    draw_rounded_rect(img, (x1, y1), (x2, y2), CARD_BG, radius=16)
    draw_rounded_rect(img, (x1, y1), (x2, y2), (60, 45, 30), radius=16, thickness=1)

    if game_state == "waiting":
        dots = "." * (1 + int(time.time() * 2) % 3)
        draw_rounded_rect(img, (x1, y1), (x1 + 6, y2), TEXT_MUTED, radius=3)
        cv2.putText(img, "LOBBY", (x1 + 26, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(img, f"Waiting for both players{dots}", (x1 + 26, y1 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, TEXT_WHITE, 2, cv2.LINE_AA)
    elif game_state == "countdown":
        remaining = COUNTDOWN_SECONDS - (time.time() - countdown_start_time)
        count_val = max(1, int(np.ceil(remaining)))
        draw_rounded_rect(img, (x1, y1), (x1 + 6, y2), ACCENT_WIN, radius=3)
        cv2.putText(img, "GET READY", (x1 + 26, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, ACCENT_WIN, 1, cv2.LINE_AA)
        cv2.putText(img, f"Match starts in {count_val}...", (x1 + 26, y1 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, TEXT_WHITE, 2, cv2.LINE_AA)
    elif game_state == "playing":
        accent = PLAYER_ACCENT[current_player]
        draw_rounded_rect(img, (x1, y1), (x1 + 6, y2), accent, radius=3)
        cv2.putText(img, "CURRENT TURN", (x1 + 26, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, accent, 1, cv2.LINE_AA)
        cv2.putText(img, PLAYER_DISPLAY_NAME[current_player], (x1 + 26, y1 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, TEXT_WHITE, 2, cv2.LINE_AA)
        if scores[current_player] == MATCH_TARGET - 1:
            draw_pill_badge(img, x2 - 130, y1 + 12, "MATCH POINT", accent)
    elif game_state == "won":
        accent = PLAYER_ACCENT[winner]
        draw_rounded_rect(img, (x1, y1), (x1 + 6, y2), ACCENT_WIN, radius=3)
        if match_complete:
            cv2.putText(img, "MATCH OVER", (x1 + 26, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, ACCENT_WIN, 1, cv2.LINE_AA)
            cv2.putText(img, f"{PLAYER_DISPLAY_NAME[match_winner]} wins the match! Press 'm' for a new match",
                        (x1 + 26, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, accent, 2, cv2.LINE_AA)
        else:
            cv2.putText(img, "ROUND OVER", (x1 + 26, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, ACCENT_WIN, 1, cv2.LINE_AA)
            cv2.putText(img, f"{PLAYER_DISPLAY_NAME[winner]} wins the round! Press 'r' to continue",
                        (x1 + 26, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, accent, 2, cv2.LINE_AA)
    else:  # draw
        draw_rounded_rect(img, (x1, y1), (x1 + 6, y2), TEXT_MUTED, radius=3)
        cv2.putText(img, "ROUND OVER", (x1 + 26, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(img, "It's a draw! Press 'r' to continue", (x1 + 26, y1 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, TEXT_WHITE, 2, cv2.LINE_AA)

    score_text = f"ROUND {rounds_played}   X: {scores['X']}   O: {scores['O']}   (first to {MATCH_TARGET})"
    text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
    cv2.putText(img, score_text, (x2 - text_size[0] - 26, y1 + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_WHITE, 1, cv2.LINE_AA)


def draw_mark_x(img, start, end, color, thickness=8):
    pad = 32
    x1, y1 = start[0] + pad, start[1] + pad
    x2, y2 = end[0] - pad, end[1] - pad
    cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x1, y2), color, thickness, cv2.LINE_AA)


def draw_mark_o(img, start, end, color, thickness=8):
    cx = (start[0] + end[0]) // 2
    cy = (start[1] + end[1]) // 2
    radius = (end[0] - start[0]) // 2 - 32
    cv2.circle(img, (cx, cy), radius, color, thickness, cv2.LINE_AA)


def draw_cell(img, cell, progress, is_winning):
    start, end = cell["start"], cell["end"]
    r, c = cell["row"], cell["col"]
    value = board[r][c]

    if is_winning:
        fill_color = lerp_color(CARD_BG, ACCENT_WIN, 0.25)
        border_color = ACCENT_WIN
    elif progress > 0 and value is None:
        fill_color = CARD_BG_HOVER
        border_color = PLAYER_ACCENT[current_player]
    else:
        fill_color = CARD_BG
        border_color = CARD_BORDER

    draw_rounded_rect(img, start, end, fill_color, radius=CORNER_RADIUS)
    draw_rounded_rect(img, start, end, border_color, radius=CORNER_RADIUS, thickness=2)

    if value == "X":
        draw_mark_x(img, start, end, ACCENT_X)
    elif value == "O":
        draw_mark_o(img, start, end, ACCENT_O)
    elif 0 < progress < 1:
        bar_y = end[1] - 8
        bar_w = int((end[0] - start[0] - 16) * progress)
        cv2.rectangle(img, (start[0] + 8, bar_y), (start[0] + 8 + bar_w, bar_y + 4),
                       PLAYER_ACCENT[current_player], -1)


def draw_help_overlay(img):
    lines = [
        ("r", "Restart the current round (keeps match score)"),
        ("m", "Start a brand new match (resets match score)"),
        ("h", "Show / hide this help panel"),
        ("q", "Quit the game"),
    ]
    x1, y1 = FRAME_W // 2 - 260, FRAME_H // 2 - 150
    x2, y2 = FRAME_W // 2 + 260, FRAME_H // 2 + 150
    overlay = img.copy()
    draw_rounded_rect(overlay, (x1, y1), (x2, y2), (10, 8, 6), radius=18)
    cv2.addWeighted(overlay, 0.9, img, 0.1, 0, img)
    draw_rounded_rect(img, (x1, y1), (x2, y2), CARD_BORDER, radius=18, thickness=2)

    cv2.putText(img, "KEYBOARD SHORTCUTS", (x1 + 30, y1 + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_WHITE, 2, cv2.LINE_AA)
    for i, (key, desc) in enumerate(lines):
        row_y = y1 + 90 + i * 42
        draw_rounded_rect(img, (x1 + 30, row_y - 22), (x1 + 70, row_y + 8), CARD_BG_HOVER, radius=8)
        cv2.putText(img, key, (x1 + 42, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ACCENT_O, 2, cv2.LINE_AA)
        cv2.putText(img, desc, (x1 + 84, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, TEXT_MUTED, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------
# 6b) Create a resizable window sized/centered to fit the screen
# ---------------------------------------------------------------------
WINDOW_NAME = "ICA Academy - Air Tic Tac Toe"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, FRAME_W, FRAME_H)
try:
    cv2.moveWindow(WINDOW_NAME, max(0, (SCREEN_W - FRAME_W) // 2), max(0, (SCREEN_H - FRAME_H) // 2))
except Exception:
    pass

# ---------------------------------------------------------------------
# 7) Main loop
# ---------------------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("WARNING: Lost the camera feed. Exiting.")
        break

    now = time.time()
    dt = now - _prev_frame_time
    dt = dt if dt > 0 else 1e-6
    fps = (fps * 0.9) + (0.1 * (1.0 / dt))
    _prev_frame_time = now

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (FRAME_W, FRAME_H))

    overlay = np.full_like(frame, BG_DARK, dtype=np.uint8)
    frame = cv2.addWeighted(frame, 0.35, overlay, 0.65, 0)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms = int(time.time() * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    current_time = time.time()

    # Collect one (smoothed) fingertip position per player for this frame
    hand_positions = {"X": None, "O": None}
    detected["X"] = False
    detected["O"] = False
    if result.hand_landmarks and result.handedness:
        for hand, handed in zip(result.hand_landmarks, result.handedness):
            label = handed[0].category_name
            player = LABEL_TO_PLAYER.get(label)
            if player is None:
                continue
            index_tip = hand[8]
            hx = int(index_tip.x * FRAME_W)
            hy = int(index_tip.y * FRAME_H)

            if smoothed_pos[player] is None:
                smoothed_pos[player] = (hx, hy)
            else:
                px, py = smoothed_pos[player]
                nx = px + (hx - px) * CURSOR_SMOOTHING
                ny = py + (hy - py) * CURSOR_SMOOTHING
                smoothed_pos[player] = (nx, ny)

            hand_positions[player] = (int(smoothed_pos[player][0]), int(smoothed_pos[player][1]))
            detected[player] = True

    # ---- state machine transitions -----------------------------------
    if game_state == "waiting" and detected["X"] and detected["O"]:
        game_state = "countdown"
        countdown_start_time = current_time

    elif game_state == "countdown":
        if not (detected["X"] and detected["O"]):
            # a player left before the countdown finished - go back to lobby
            game_state = "waiting"
            countdown_start_time = None
        elif current_time - countdown_start_time >= COUNTDOWN_SECONDS:
            game_state = "playing"

    draw_header(frame)
    draw_detection_panel(frame)
    draw_status_panel(frame)

    # Figure out which cell (if any) is being hovered by the active player
    active_cell = None
    active_progress = 0.0
    if game_state == "playing":
        pos = hand_positions[current_player]
        if pos is not None:
            px, py = pos
            for r in range(3):
                for c in range(3):
                    cell = cells[r][c]
                    sx, sy = cell["start"]
                    ex, ey = cell["end"]
                    if sx < px < ex and sy < py < ey and board[r][c] is None:
                        active_cell = (r, c)

        if active_cell != hover_cell[current_player]:
            hover_cell[current_player] = active_cell
            hover_start[current_player] = current_time if active_cell else None

        if active_cell is not None:
            elapsed = current_time - hover_start[current_player]
            active_progress = min(elapsed / threshold_time, 1.0)
            if elapsed >= threshold_time:
                r, c = active_cell
                board[r][c] = current_player
                hover_cell[current_player] = None
                hover_start[current_player] = None

                win_player, win_line = check_winner(board)
                if win_player:
                    game_state = "won"
                    winner = win_player
                    winning_line = win_line
                    scores[win_player] += 1
                    rounds_played += 1
                    celebration_start_time = current_time
                    line_cells = [cells[rr][cc] for (rr, cc) in win_line]
                    cx = sum(c["start"][0] + c["end"][0] for c in line_cells) // (2 * len(line_cells))
                    cy = sum(c["start"][1] + c["end"][1] for c in line_cells) // (2 * len(line_cells))
                    spawn_particles((cx, cy), PLAYER_ACCENT[win_player])
                    if scores[win_player] >= MATCH_TARGET:
                        match_complete = True
                        match_winner = win_player
                elif is_full(board):
                    game_state = "draw"
                    rounds_played += 1
                else:
                    current_player = "O" if current_player == "X" else "X"

    # Draw the board
    for r in range(3):
        for c in range(3):
            cell = cells[r][c]
            is_winning = winning_line is not None and (r, c) in winning_line
            progress = active_progress if active_cell == (r, c) else 0.0
            draw_cell(frame, cell, progress, is_winning)

    if celebration_start_time is not None:
        update_particles(dt)
        draw_particles(frame)

    # Draw both players' finger cursors (their own accent color)
    for player, pos in hand_positions.items():
        if pos is None:
            continue
        px, py = pos
        accent = PLAYER_ACCENT[player]
        dim = accent if player == current_player or game_state != "playing" else CARD_BORDER
        cv2.circle(frame, (px, py), 18, dim, 2, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 8, dim, cv2.FILLED, cv2.LINE_AA)
        cv2.putText(frame, PLAYER_DISPLAY_NAME[player].split(" - ")[0], (px + 22, py - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, dim, 2, cv2.LINE_AA)

    footer_text = "r: restart round | m: new match | h: help | q: quit"
    cv2.putText(frame, footer_text, (90, FRAME_H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_MUTED, 1, cv2.LINE_AA)

    if show_help:
        draw_help_overlay(frame)

    cv2.imshow(WINDOW_NAME, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        reset_round(keep_scores=True)
    elif key == ord("m"):
        start_new_match()
    elif key == ord("h"):
        show_help = not show_help

cap.release()
cv2.destroyAllWindows()
landmarker.close()

print("\n--- Session summary ---")
print(f"Rounds played : {rounds_played}")
print(f"Final score   : Player 1 (X) {scores['X']}  -  Player 2 (O) {scores['O']}")
if match_complete and match_winner:
    print(f"Match winner  : {PLAYER_DISPLAY_NAME[match_winner]}")
print("Thanks for playing - ICA Academy")

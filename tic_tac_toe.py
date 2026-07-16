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
      Left hand  -> Player X (blue accent)
      Right hand -> Player O (cyan accent)
  so it doesn't matter which hand shows up first in a given frame, or
  if a hand briefly leaves the frame and comes back.
- The game is still turn-based (classic Tic Tac Toe rules), but since
  both hands are tracked live, players don't have to take turns handing
  over a mouse/keyboard - whoever's turn it is just points at a cell
  and holds their finger there to place their mark.
- Point your index finger at an empty cell and hold it for half a
  second to place your mark. Press 'r' to start a new round (keeps the
  score), or 'q' to quit.
"""

import os
import time
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

ACCENT_X = (255, 108, 59)    # Player X - "blue" end of the ICA gradient
ACCENT_O = (238, 211, 34)    # Player O - "cyan" end of the ICA gradient
ACCENT_WIN = (90, 214, 120)  # highlight color for the winning line

CORNER_RADIUS = 16
threshold_time = 0.5
FRAME_W, FRAME_H = 1440, 900

LABEL_TO_PLAYER = {"Left": "X", "Right": "O"}
PLAYER_ACCENT = {"X": ACCENT_X, "O": ACCENT_O}


def lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


# ---------------------------------------------------------------------
# 4) Board layout - 3x3 grid, centered under the header/status panel
# ---------------------------------------------------------------------
CELL_SIZE = 180
CELL_GAP = 16
BOARD_SIZE = CELL_SIZE * 3 + CELL_GAP * 2
BOARD_X = (FRAME_W - BOARD_SIZE) // 2
BOARD_Y = 250

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
game_state = "playing"     # "playing" | "won" | "draw"
winner = None
winning_line = None
scores = {"X": 0, "O": 0}

hover_cell = {"X": None, "O": None}
hover_start = {"X": None, "O": None}


def check_winner(b):
    for line in WIN_LINES:
        vals = [b[r][c] for (r, c) in line]
        if vals[0] is not None and vals[0] == vals[1] == vals[2]:
            return vals[0], line
    return None, None


def is_full(b):
    return all(b[r][c] is not None for r in range(3) for c in range(3))


def reset_round():
    global board, current_player, game_state, winner, winning_line
    global hover_cell, hover_start
    board = new_board()
    current_player = "X"
    game_state = "playing"
    winner = None
    winning_line = None
    hover_cell = {"X": None, "O": None}
    hover_start = {"X": None, "O": None}


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

    cv2.putText(img, "Air Tic Tac Toe", (90, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, TEXT_WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, "Left hand = Player X   |   Right hand = Player O   |   hold finger on a cell to play",
                (90, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_MUTED, 1, cv2.LINE_AA)


def draw_status_panel(img):
    x1, y1, x2, y2 = 90, 150, FRAME_W - 90, 220
    draw_rounded_rect(img, (x1, y1), (x2, y2), CARD_BG, radius=16)
    draw_rounded_rect(img, (x1, y1), (x2, y2), (60, 45, 30), radius=16, thickness=1)

    if game_state == "playing":
        accent = PLAYER_ACCENT[current_player]
        draw_rounded_rect(img, (x1, y1), (x1 + 6, y2), accent, radius=3)
        cv2.putText(img, "CURRENT TURN", (x1 + 26, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, accent, 1, cv2.LINE_AA)
        cv2.putText(img, f"Player {current_player}", (x1 + 26, y1 + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_WHITE, 2, cv2.LINE_AA)
    elif game_state == "won":
        accent = PLAYER_ACCENT[winner]
        draw_rounded_rect(img, (x1, y1), (x1 + 6, y2), ACCENT_WIN, radius=3)
        cv2.putText(img, "GAME OVER", (x1 + 26, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, ACCENT_WIN, 1, cv2.LINE_AA)
        cv2.putText(img, f"Player {winner} wins!  Press 'r' to play again", (x1 + 26, y1 + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, accent, 2, cv2.LINE_AA)
    else:  # draw
        draw_rounded_rect(img, (x1, y1), (x1 + 6, y2), TEXT_MUTED, radius=3)
        cv2.putText(img, "GAME OVER", (x1 + 26, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(img, "It's a draw!  Press 'r' to play again", (x1 + 26, y1 + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_WHITE, 2, cv2.LINE_AA)

    score_text = f"X: {scores['X']}    O: {scores['O']}"
    text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.putText(img, score_text, (x2 - text_size[0] - 26, y1 + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_WHITE, 2, cv2.LINE_AA)


def draw_mark_x(img, start, end, color, thickness=8):
    pad = 34
    x1, y1 = start[0] + pad, start[1] + pad
    x2, y2 = end[0] - pad, end[1] - pad
    cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x1, y2), color, thickness, cv2.LINE_AA)


def draw_mark_o(img, start, end, color, thickness=8):
    cx = (start[0] + end[0]) // 2
    cy = (start[1] + end[1]) // 2
    radius = (end[0] - start[0]) // 2 - 34
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


# ---------------------------------------------------------------------
# 7) Main loop
# ---------------------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (FRAME_W, FRAME_H))

    overlay = np.full_like(frame, BG_DARK, dtype=np.uint8)
    frame = cv2.addWeighted(frame, 0.35, overlay, 0.65, 0)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms = int(time.time() * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    current_time = time.time()

    # Collect one fingertip position per player for this frame
    hand_positions = {"X": None, "O": None}
    if result.hand_landmarks and result.handedness:
        for hand, handed in zip(result.hand_landmarks, result.handedness):
            label = handed[0].category_name
            player = LABEL_TO_PLAYER.get(label)
            if player is None:
                continue
            index_tip = hand[8]
            hx = int(index_tip.x * FRAME_W)
            hy = int(index_tip.y * FRAME_H)
            hand_positions[player] = (hx, hy)

    draw_header(frame)
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
                elif is_full(board):
                    game_state = "draw"
                else:
                    current_player = "O" if current_player == "X" else "X"

    # Draw the board
    for r in range(3):
        for c in range(3):
            cell = cells[r][c]
            is_winning = winning_line is not None and (r, c) in winning_line
            progress = active_progress if active_cell == (r, c) else 0.0
            draw_cell(frame, cell, progress, is_winning)

    # Draw both players' finger cursors (their own accent color)
    for player, pos in hand_positions.items():
        if pos is None:
            continue
        px, py = pos
        accent = PLAYER_ACCENT[player]
        dim = accent if player == current_player or game_state != "playing" else CARD_BORDER
        cv2.circle(frame, (px, py), 18, dim, 2)
        cv2.circle(frame, (px, py), 8, dim, cv2.FILLED)
        cv2.putText(frame, player, (px + 22, py - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, dim, 2, cv2.LINE_AA)

    cv2.putText(frame, "Press 'r' to restart the round  |  'q' to quit", (90, FRAME_H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_MUTED, 1, cv2.LINE_AA)

    cv2.imshow("ICA Academy - Air Tic Tac Toe", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        reset_round()

cap.release()
cv2.destroyAllWindows()
landmarker.close()
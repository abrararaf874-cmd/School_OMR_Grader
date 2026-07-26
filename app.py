"""
🏫 Feni Model High School — OMR Grader v4.0
Features: Auto answer key extraction | Regrade without rescan | Per-question CSV | Item analysis
"""

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import io
import pandas as pd
import csv
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Feni Model HS OMR Grader",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ═══════════════════════════════════════════════════════════════
# SAFE RERUN
# ═══════════════════════════════════════════════════════════════
def safe_rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


# ═══════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+Bengali:wght@400;500;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', 'Noto Sans Bengali', sans-serif; }
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e40af 50%, #0f766e 100%);
        padding: 2rem; border-radius: 20px; text-align: center;
        color: white; margin-bottom: 1.5rem;
        box-shadow: 0 20px 25px -5px rgba(0,0,0,0.15);
    }
    .hero h1 { margin: 0; font-size: 1.8rem; font-weight: 700; }
    .hero h3 { margin: 0.4rem 0 0 0; font-weight: 400; opacity: 0.9; font-size: 1rem; }
    .hero .bn { font-family: 'Noto Sans Bengali', sans-serif; margin-top: 0.3rem; opacity: 0.85; font-size: 0.95rem; }
    .answer-box {
        background: #f0f9ff; border: 2px solid #0ea5e9; border-radius: 16px;
        padding: 1.25rem; margin-bottom: 1.5rem;
    }
    .answer-box h4 { color: #0369a1; margin: 0 0 0.5rem 0; }
    .key-preview { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.75rem; }
    .key-chip {
        background: #e0f2fe; color: #0369a1; padding: 0.25rem 0.6rem;
        border-radius: 8px; font-size: 0.8rem; font-weight: 600; border: 1px solid #7dd3fc;
    }
    .key-chip.multi { background: #fef3c7; color: #92400e; border-color: #fcd34d; }
    .key-chip.bonus { background: #d1fae5; color: #065f46; border-color: #6ee7b7; }
    .key-chip.blank { background: #fee2e2; color: #991b1b; border-color: #fca5a5; }
    .mode-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem; }
    @media (max-width: 768px) { .mode-grid { grid-template-columns: 1fr; } }
    .mode-card {
        background: white; border: 2px solid #e2e8f0; border-radius: 16px;
        padding: 1.5rem; text-align: center; cursor: pointer; transition: all 0.2s;
    }
    .mode-card:hover { border-color: #3b82f6; transform: translateY(-2px); }
    .mode-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
    .mode-title { font-size: 1.1rem; font-weight: 700; color: #1e293b; }
    .mode-desc { color: #64748b; font-size: 0.85rem; margin-top: 0.3rem; }
    .camera-gate {
        background: #f8fafc; border: 3px dashed #cbd5e1; border-radius: 20px;
        padding: 3rem 1.5rem; text-align: center; margin: 1.5rem 0;
    }
    .camera-gate .icon { font-size: 3rem; margin-bottom: 0.5rem; }
    .score-card {
        background: white; border-radius: 16px; padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        border-left: 5px solid #10b981; margin-bottom: 1rem;
    }
    .score-card.fail { border-left-color: #ef4444; }
    .score-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; }
    .score-filename { font-weight: 600; color: #334155; font-size: 0.9rem; word-break: break-all; }
    .score-badge { padding: 0.25rem 0.75rem; border-radius: 999px; font-size: 0.75rem; font-weight: 700; }
    .badge-pass { background: #d1fae5; color: #065f46; }
    .badge-fail { background: #fee2e2; color: #991b1b; }
    .score-big { font-size: 2.5rem; font-weight: 800; color: #0f172a; }
    .score-label { color: #64748b; font-size: 0.85rem; }
    .score-bar-bg { background: #e2e8f0; height: 6px; border-radius: 999px; margin-top: 0.75rem; overflow: hidden; }
    .score-bar-fill { height: 100%; border-radius: 999px; }
    .item-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
    .item-table th { background: #f1f5f9; padding: 0.6rem; text-align: center; border-bottom: 2px solid #e2e8f0; }
    .item-table td { padding: 0.5rem; text-align: center; border-bottom: 1px solid #f1f5f9; }
    .item-table tr:hover { background: #f8fafc; }
    .diff-easy { color: #059669; font-weight: 600; }
    .diff-med { color: #d97706; font-weight: 600; }
    .diff-hard { color: #dc2626; font-weight: 600; }
    div[data-testid="stButton"] > button { border-radius: 10px !important; font-weight: 600 !important; }
    .instruction-card { background: #f8fafc; border-radius: 12px; padding: 1.25rem; margin-top: 1rem; }
    .instruction-card h4 { color: #1e40af; margin-top: 0; }
    .app-footer { text-align: center; padding: 1.5rem; color: #94a3b8; font-size: 0.8rem; margin-top: 2rem; border-top: 1px solid #e2e8f0; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════
defaults = {
    'camera_active': False, 'mode': None, 'processed_results': [],
    'detailed_results': [], 'language': 'English', 'debug_mode': False,
    'passing_score': 33.0, 'fill_threshold': 0.35,
    'last_answer_key_str': '', 'master_cam_active': False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "main_answer_key_input" not in st.session_state:
    st.session_state.main_answer_key_input = "B, A, C, X, D, A/B, B, C, A, D, B, C, A, D, B, X, C, A, D, B, C, A, D, B, C"


# ═══════════════════════════════════════════════════════════════
# OMR CONFIG
# ═══════════════════════════════════════════════════════════════
OPTIONS = ["A", "B", "C", "D"]
WARPED_SIZE = (850, 1100)
COLUMN_ROIS_REL = [
    (0.40, 0.52, 0.19, 0.44),
    (0.59, 0.52, 0.19, 0.44),
    (0.78, 0.52, 0.19, 0.44),
]
MIN_BUBBLE_DIM = 10
MAX_BUBBLE_DIM = 45
MULTI_MARK_RELATIVE_MARGIN = 0.28
MIN_CIRCULARITY = 0.65


def get_absolute_rois(width, height):
    return [(int(x * width), int(y * height), int(w * width), int(h * height))
            for x, y, w, h in COLUMN_ROIS_REL]


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts, out_size):
    rect = order_points(pts)
    w, h = out_size
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (w, h))


def find_sheet_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    img_area = image.shape[0] * image.shape[1]
    candidates = []
    for thresh_type in [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV]:
        _, mask = cv2.threshold(blurred, 0, 255, thresh_type + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            largest = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(largest) >= 0.08 * img_area:
                peri = cv2.arcLength(largest, True)
                approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
                if len(approx) == 4:
                    return approx.reshape(4, 2).astype("float32")
                candidates.append(largest)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        largest = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(largest) >= 0.08 * img_area:
            peri = cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype("float32")
            candidates.append(largest)
    if candidates:
        best = max(candidates, key=cv2.contourArea)
        rect = cv2.minAreaRect(best)
        return cv2.boxPoints(rect).astype("float32")
    return None


def is_valid_bubble(c):
    x, y, w, h = cv2.boundingRect(c)
    if not (MIN_BUBBLE_DIM <= w <= MAX_BUBBLE_DIM and MIN_BUBBLE_DIM <= h <= MAX_BUBBLE_DIM):
        return False
    if not (0.7 <= w / float(h) <= 1.3):
        return False
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    if peri == 0:
        return False
    circularity = 4 * np.pi * area / (peri ** 2)
    if circularity < MIN_CIRCULARITY:
        return False
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return False
    solidity = area / hull_area
    return solidity >= 0.85


def cluster_bubbles_into_rows(bubbles):
    if not bubbles:
        return []
    heights = [b[1][3] for b in bubbles]
    median_h = float(np.median(heights)) if heights else 20.0
    y_tol = max(median_h * 0.6, 12.0)
    bubbles = sorted(bubbles, key=lambda b: b[1][1])
    rows = []
    current = [bubbles[0]]
    for bubble in bubbles[1:]:
        _, (_, y, _, _) = bubble
        mean_y = np.mean([b[1][1] for b in current])
        if abs(y - mean_y) <= y_tol:
            current.append(bubble)
        else:
            current = sorted(current, key=lambda b: b[1][0])
            if len(current) >= 3:
                rows.append(current)
            current = [bubble]
    current = sorted(current, key=lambda b: b[1][0])
    if len(current) >= 3:
        rows.append(current)
    return rows


def assign_bubbles_to_slots(row, roi_width):
    slot_width = roi_width / 4.0
    slot_centers = [slot_width * i + slot_width / 2.0 for i in range(4)]
    row = sorted(row, key=lambda b: b[1][0])
    assignments = [None] * 4
    used = [False] * len(row)
    for slot_idx, slot_x in enumerate(slot_centers):
        best_dist = float('inf')
        best_b = -1
        for b_idx, (c, (x, y, w, h)) in enumerate(row):
            if used[b_idx]:
                continue
            cx = x + w / 2.0
            dist = abs(cx - slot_x)
            if dist < best_dist and dist < slot_width * 0.7:
                best_dist = dist
                best_b = b_idx
        if best_b >= 0:
            used[best_b] = True
            assignments[slot_idx] = row[best_b][0]
    return assignments


# ═══════════════════════════════════════════════════════════════
# ANSWER KEY PARSING & STRINGIFY
# ═══════════════════════════════════════════════════════════════
def parse_answer_key(raw_string):
    parts = [p.strip().upper() for p in raw_string.split(",") if p.strip()]
    parsed = []
    for p in parts:
        if p in ("X", "NONE", "NA", "-", "*", "SKIP"):
            parsed.append(set())
        else:
            chars = set()
            for ch in p:
                if ch in OPTIONS:
                    chars.add(ch)
            for segment in p.split("/"):
                segment = segment.strip()
                if segment in OPTIONS:
                    chars.add(segment)
            parsed.append(chars)
    return parsed


def answer_key_to_string(answer_key):
    parts = []
    for s in answer_key:
        if len(s) == 0:
            parts.append("X")
        elif len(s) == 1:
            parts.append(list(s)[0])
        else:
            parts.append("/".join(sorted(s)))
    return ", ".join(parts)


# ═══════════════════════════════════════════════════════════════
# COLUMN PROCESSING — GRADING MODE
# ═══════════════════════════════════════════════════════════════
def process_column(roi_thresh, roi_color, start_q, answer_key, options, fill_thresh, debug=False):
    cnts, _ = cv2.findContours(roi_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    bubbles = [(c, cv2.boundingRect(c)) for c in cnts if is_valid_bubble(c)]
    results = []
    correct = 0
    roi_h, roi_w = roi_thresh.shape

    if len(bubbles) < 12:
        if debug:
            st.warning(f"Col Q{start_q}: Only {len(bubbles)} bubbles found.")
        return results, 0

    rows = cluster_bubbles_into_rows(bubbles)
    if len(rows) < 3:
        if debug:
            st.warning(f"Col Q{start_q}: Too few rows ({len(rows)}).")
        return results, 0

    for r_idx, row in enumerate(rows):
        q_num = start_q + r_idx
        if q_num > len(answer_key):
            break
        if len(row) < 3:
            continue

        row = sorted(row, key=lambda b: b[1][0])
        if len(row) > 4:
            row = sorted(row, key=lambda b: -cv2.contourArea(b[0]))[:4]
            row = sorted(row, key=lambda b: b[1][0])

        slot_contours = assign_bubbles_to_slots(row, roi_w)
        fills = []
        for slot_idx, c in enumerate(slot_contours):
            if c is None:
                fills.append(0.0)
                continue
            mask = np.zeros(roi_thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.erode(mask, np.ones((2, 2), np.uint8), iterations=1)
            area = cv2.countNonZero(mask)
            if area == 0:
                fills.append(0.0)
                continue
            filled = cv2.countNonZero(cv2.bitwise_and(roi_thresh, roi_thresh, mask=mask))
            fills.append(filled / float(area))

        while len(fills) < 4:
            fills.append(0.0)

        best_idx = int(np.argmax(fills))
        best_val = fills[best_idx]
        runner = sorted(fills, reverse=True)[1] if len(fills) > 1 else 0.0

        if best_val < fill_thresh:
            marked = None
        elif runner > 0 and ((best_val - runner) / best_val) < MULTI_MARK_RELATIVE_MARGIN:
            marked = "MULTI"
        else:
            marked = options[best_idx]

        correct_set = answer_key[q_num - 1]
        if len(correct_set) == 0:
            is_correct = True
            correct_letter_display = "✓ (Any)"
        else:
            is_correct = (marked in correct_set)
            correct_letter_display = "/".join(sorted(correct_set))

        correct += int(is_correct)
        results.append({
            "question": q_num, "marked": marked,
            "correct": correct_letter_display, "is_correct": is_correct,
        })

        for slot_idx, c in enumerate(slot_contours):
            if c is None:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w // 2, y + h // 2
            r = max(w, h) // 2 + 1
            letter = options[slot_idx]
            if letter == marked and is_correct:
                cv2.circle(roi_color, (cx, cy), r, (0, 255, 0), 3)
            elif letter == marked and not is_correct:
                cv2.circle(roi_color, (cx, cy), r, (0, 0, 255), 3)
            if letter in correct_set and not is_correct:
                cv2.circle(roi_color, (cx, cy), r, (0, 255, 0), 1)

    return results, correct


# ═══════════════════════════════════════════════════════════════
# COLUMN PROCESSING — EXTRACTION MODE (for master sheet)
# ═══════════════════════════════════════════════════════════════
def extract_answer_key_column(roi_thresh, roi_color, start_q, options, fill_thresh, debug=False):
    cnts, _ = cv2.findContours(roi_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    bubbles = [(c, cv2.boundingRect(c)) for c in cnts if is_valid_bubble(c)]
    extracted_sets = []
    roi_h, roi_w = roi_thresh.shape

    if len(bubbles) < 12:
        return extracted_sets

    rows = cluster_bubbles_into_rows(bubbles)
    if len(rows) < 3:
        return extracted_sets

    for r_idx, row in enumerate(rows):
        q_num = start_q + r_idx
        if len(row) < 3:
            continue
        row = sorted(row, key=lambda b: b[1][0])
        if len(row) > 4:
            row = sorted(row, key=lambda b: -cv2.contourArea(b[0]))[:4]
            row = sorted(row, key=lambda b: b[1][0])

        slot_contours = assign_bubbles_to_slots(row, roi_w)
        fills = []
        for slot_idx, c in enumerate(slot_contours):
            if c is None:
                fills.append(0.0)
                continue
            mask = np.zeros(roi_thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.erode(mask, np.ones((2, 2), np.uint8), iterations=1)
            area = cv2.countNonZero(mask)
            if area == 0:
                fills.append(0.0)
                continue
            filled = cv2.countNonZero(cv2.bitwise_and(roi_thresh, roi_thresh, mask=mask))
            fills.append(filled / float(area))

        while len(fills) < 4:
            fills.append(0.0)

        best_idx = int(np.argmax(fills))
        best_val = fills[best_idx]
        runner = sorted(fills, reverse=True)[1] if len(fills) > 1 else 0.0

        if best_val < fill_thresh * 0.6:
            detected = set()
        elif runner > 0 and ((best_val - runner) / best_val) < MULTI_MARK_RELATIVE_MARGIN * 1.5:
            top2 = np.argsort(fills)[-2:]
            detected = {options[i] for i in top2 if fills[i] > fill_thresh * 0.4}
        else:
            detected = {options[best_idx]}

        extracted_sets.append(detected)

        for slot_idx, c in enumerate(slot_contours):
            if c is None:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w // 2, y + h // 2
            r = max(w, h) // 2 + 1
            letter = options[slot_idx]
            if letter in detected:
                cv2.circle(roi_color, (cx, cy), r, (0, 255, 0), 3)
                cv2.putText(roi_color, letter, (cx - 5, cy + 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    return extracted_sets


# ═══════════════════════════════════════════════════════════════
# IMAGE PROCESSING — GRADING
# ═══════════════════════════════════════════════════════════════
def process_single_image(image_bytes, answer_key, num_questions, fill_thresh, debug=False):
    pil_img = Image.open(io.BytesIO(image_bytes))
    pil_img = ImageOps.exif_transpose(pil_img)
    pil_img = pil_img.convert("RGB")
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if img is None:
        return None, 0, 0, "Could not read image", []

    max_dim = 2000
    h, w = img

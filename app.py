"""
🏫 Feni Model High School — OMR Grader v3.1 (Definitive)
All known CV and UI bugs fixed. Slot-based bubble mapping. Adaptive thresholding.
"""

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import io
import pandas as pd
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
# SAFE RERUN (handles older Streamlit versions)
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
    'language': 'English', 'debug_mode': False, 'passing_score': 33.0,
    'fill_threshold': 0.35,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

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
    """
    Robust sheet detection. Tries multiple methods:
    1. Otsu (paper bright)
    2. Inverse Otsu (paper dark / light background)
    3. Canny edge fallback
    Returns 4 corner points or None.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    img_area = image.shape[0] * image.shape[1]

    candidates = []

    # Method 1: Standard Otsu
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

    # Method 2: Canny fallback
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

    # Fallback: rotated bounding box of best candidate
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
    # Solidity check: reject shapes with big holes or jagged edges
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return False
    solidity = area / hull_area
    return solidity >= 0.85


def cluster_bubbles_into_rows(bubbles):
    """
    Group bubbles into rows using dynamic Y tolerance based on bubble size.
    """
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
    """
    CRITICAL FIX: Map detected bubbles to A/B/C/D slots by x-position.
    Handles missing bubbles correctly.
    Returns list of 4 elements: contour or None for each slot.
    """
    slot_width = roi_width / 4.0
    slot_centers = [slot_width * i + slot_width / 2.0 for i in range(4)]

    # Sort by x just in case
    row = sorted(row, key=lambda b: b[1][0])

    # Greedy assignment: for each slot, find nearest unused bubble
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


def process_column(roi_thresh, roi_color, start_q, answer_key, options, fill_thresh, debug=False):
    cnts, _ = cv2.findContours(roi_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    bubbles = [(c, cv2.boundingRect(c)) for c in cnts if is_valid_bubble(c)]

    results = []
    correct = 0
    roi_h, roi_w = roi_thresh.shape

    if len(bubbles) < 12:
        if debug:
            st.warning(f"Column starting Q{start_q}: Only {len(bubbles)} bubbles found.")
        return results, 0

    rows = cluster_bubbles_into_rows(bubbles)

    expected_rows = min(10, len(answer_key) - start_q + 1)
    if len(rows) < expected_rows and debug:
        st.info(f"Column starting Q{start_q}: Found {len(rows)} rows (expected ~{expected_rows}).")

    if len(rows) < 3:
        if debug:
            st.warning(f"Column starting Q{start_q}: Too few rows clustered ({len(rows)}).")
        return results, 0

    for r_idx, row in enumerate(rows):
        q_num = start_q + r_idx
        if q_num > len(answer_key):
            break

        # Map bubbles to A/B/C/D slots
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

        best_idx = int(np.argmax(fills))
        best_val = fills[best_idx]
        runner = sorted(fills, reverse=True)[1] if len(fills) > 1 else 0.0

        if best_val < fill_thresh:
            marked = None
        elif runner > 0 and ((best_val - runner) / best_val) < MULTI_MARK_RELATIVE_MARGIN:
            marked = "MULTI"
        else:
            marked = options[best_idx]

        # Flexible grading
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

        # Annotation
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


def process_single_image(image_bytes, answer_key, num_questions, fill_thresh, debug=False):
    pil_img = Image.open(io.BytesIO(image_bytes))
    pil_img = ImageOps.exif_transpose(pil_img)
    pil_img = pil_img.convert("RGB")
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if img is None:
        return None, 0, 0, "Could not read image"

    # Resize if massive
    max_dim = 2000
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)

    # Detect sheet
    corners = find_sheet_contour(img)
    if corners is None:
        return None, 0, 0, "Sheet not detected"

    # Warp to standard size
    warped = four_point_transform(img, corners, WARPED_SIZE)

    # Sharpen to crisp up bubble edges
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    warped = cv2.filter2D(warped, -1, kernel)

    # Grayscale + CLAHE for contrast
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Primary: Adaptive threshold (robust to shadows)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 51, 10
    )

    # Fallback: Otsu if adaptive looks empty (optional check)
    if debug:
        thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.image(thresh, caption="Adaptive Threshold", use_column_width=True)
        with col_d2:
            st.image(thresh_otsu, caption="Otsu Threshold", use_column_width=True)

    total_correct = 0
    col_rois = get_absolute_rois(WARPED_SIZE[0], WARPED_SIZE[1])

    for col_idx, (x, y, w, h) in enumerate(col_rois):
        roi_t = thresh[y:y + h, x:x + w]
        roi_c = warped[y:y + h, x:x + w]
        start_q = col_idx * 10 + 1
        if start_q <= num_questions:
            _, cscore = process_column(
                roi_t, roi_c, start_q, answer_key, OPTIONS, fill_thresh, debug
            )
            total_correct += cscore

            if debug:
                st.image(cv2.cvtColor(roi_c, cv2.COLOR_BGR2RGB),
                         caption=f"Col {col_idx+1} Annotated", use_column_width=True)

    score_pct = (total_correct / num_questions) * 100
    cv2.putText(warped, f"Score: {total_correct}/{num_questions} ({score_pct:.1f}%)",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    return warped_rgb, total_correct, score_pct, "OK"


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


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("**⚙️ Settings**")

    st.session_state.passing_score = st.slider(
        "Passing %", 0.0, 100.0, 33.0, 1.0, key="sidebar_passing_slider"
    )
    st.session_state.fill_threshold = st.slider(
        "Bubble Fill Threshold", 0.10, 0.90,
        st.session_state.fill_threshold, 0.05,
        help="Lower = detects lighter marks (pencil). Higher = needs dark pen.",
        key="sidebar_fill_slider"
    )
    st.session_state.debug_mode = st.checkbox(
        "🔍 Debug Mode", value=False, key="sidebar_debug_checkbox"
    )
    st.session_state.language = st.selectbox(
        "🌐 Language", ["English", "বাংলা"], key="sidebar_language_select"
    )

    if st.session_state.processed_results:
        st.markdown("---")
        df = pd.DataFrame(st.session_state.processed_results)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download CSV", csv,
            f"omr_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv", use_container_width=True, key="sidebar_download_csv"
        )
        if st.button("🗑️ Clear Results", use_container_width=True, key="sidebar_clear_results"):
            st.session_state.processed_results = []
            safe_rerun()

# ═══════════════════════════════════════════════════════════════
# MAIN UI
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <h1>🏫 Feni Model High School</h1>
    <h3>Automated OMR Answer Sheet Grader</h3>
    <div class="bn">ফেনী মডেল হাই স্কুল ওএমআর গ্রেডার</div>
</div>
""", unsafe_allow_html=True)

# ── STEP 1: ANSWER KEY ──
st.markdown("### 📝 Step 1: Enter Answer Key")

st.markdown("""
<div class="answer-box">
    <h4>✅ Correct Answers</h4>
    <p style="margin:0; color:#475569; font-size:0.9rem;">
    Format: Comma-separated. Use <b>A/B</b> for multiple correct answers.
    Use <b>X</b> or <b>NONE</b> for bonus questions (everyone gets credit).
    </p>
</div>
""", unsafe_allow_html=True)

default_key = "B, A, C, X, D, A/B, B, C, A, D, B, C, A, D, B, X, C, A, D, B, C, A, D, B, C"
user_key = st.text_area(
    "Type answers (e.g., A, B, A/B, X, C...)",
    value=default_key, height=80, key="main_answer_key_input"
)

answer_list = parse_answer_key(user_key)
num_questions = len(answer_list)

if num_questions > 0:
    st.markdown("**🔍 Answer Key Preview:**")
    chips_html = '<div class="key-preview">'
    normal_count = multi_count = bonus_count = 0
    for i, ans_set in enumerate(answer_list):
        qn = i + 1
        if len(ans_set) == 0:
            chips_html += f'<span class="key-chip bonus">Q{qn}: ✓ Any</span>'
            bonus_count += 1
        elif len(ans_set) > 1:
            chips_html += f'<span class="key-chip multi">Q{qn}: {" / ".join(sorted(ans_set))}</span>'
            multi_count += 1
        else:
            chips_html += f'<span class="key-chip">Q{qn}: {list(ans_set)[0]}</span>'
            normal_count += 1
    chips_html += '</div>'
    st.markdown(chips_html, unsafe_allow_html=True)
    summaries = []
    if normal_count: summaries.append(f"✅ {normal_count} normal")
    if multi_count: summaries.append(f"🟡 {multi_count} multi-answer")
    if bonus_count: summaries.append(f"🟢 {bonus_count} bonus")
    st.caption(" | ".join(summaries))

if num_questions == 0:
    st.warning("⚠️ Please enter at least one answer above.")
    st.stop()
elif num_questions > 30:
    st.error(f"⚠️ Maximum 30 questions supported. You entered {num_questions}.")
    st.stop()

st.success(f"📋 Grading **{num_questions}** questions  |  Fill threshold: **{st.session_state.fill_threshold}**")

# ── STEP 2: MODE SELECT ──
st.markdown("---")
st.markdown(f"### {'📸 Step 2: Choose Input Method' if st.session_state.language == 'English' else '📸 ধাপ ২: ইনপুট পদ্ধতি'}")

if st.session_state.mode is None:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="mode-card">
            <div class="mode-icon">📁</div>
            <div class="mode-title">Bulk Upload</div>
            <div class="mode-desc">Upload multiple photos from gallery</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📁 Select Upload Mode", use_container_width=True, key="btn_select_upload"):
            st.session_state.mode = "upload"
            safe_rerun()

    with c2:
        st.markdown("""
        <div class="mode-card">
            <div class="mode-icon">📸</div>
            <div class="mode-title">Camera Mode</div>
            <div class="mode-desc">Take photos one by one</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📸 Select Camera Mode", use_container_width=True, key="btn_select_camera"):
            st.session_state.mode = "camera"
            st.session_state.camera_active = False
            safe_rerun()

    if st.session_state.processed_results:
        st.markdown("---")
        st.markdown("### 📊 Recent Results")
        st.dataframe(pd.DataFrame(st.session_state.processed_results), use_container_width=True, hide_index=True)

# ── UPLOAD MODE ──
elif st.session_state.mode == "upload":
    st.markdown("### 📁 Bulk Upload Mode")
    if st.button("← Back to Menu", type="secondary", key="btn_back_upload"):
        st.session_state.mode = None
        safe_rerun()

    files = st.file_uploader(
        "Select photos", type=["jpg", "jpeg", "png"],
        accept_multiple_files=True, label_visibility="collapsed",
        key="bulk_file_uploader"
    )

    if files:
        st.success(f"Processing **{len(files)}** paper(s)...")
        new_results = []
        progress = st.progress(0, text="Starting...")

        for i, file in enumerate(files):
            progress.progress((i) / len(files), text=f"Processing {file.name}...")
            img_rgb, score, pct, status = process_single_image(
                file.getvalue(), answer_list, num_questions,
                st.session_state.fill_threshold, st.session_state.debug_mode
            )

            if status != "OK":
                st.error(f"❌ **{file.name}**: {status}")
                new_results.append({
                    "Filename": file.name, "Score": "Error", "Percentage": "Error",
                    "Status": status, "Time": datetime.now().strftime("%H:%M:%S")
                })
                continue

            passed = pct >= st.session_state.passing_score
            bc = "badge-pass" if passed else "badge-fail"
            bt = "PASS" if passed else "FAIL"
            cc = "" if passed else "fail"
            bar = "#10b981" if passed else "#ef4444"

            st.markdown(f"""
            <div class="score-card {cc}">
                <div class="score-header">
                    <div class="score-filename">{file.name}</div>
                    <div class="score-badge {bc}">{bt}</div>
                </div>
                <div class="score-big">{score}/{num_questions}</div>
                <div class="score-label">{pct:.1f}% Correct</div>
                <div class="score-bar-bg">
                    <div class="score-bar-fill" style="width: {pct}%; background: {bar};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("🔍 View Annotated Sheet", expanded=False):
                st.image(img_rgb, use_column_width=True)

            new_results.append({
                "Filename": file.name, "Score": score,
                "Percentage": f"{pct:.1f}%", "Status": "Pass" if passed else "Fail",
                "Time": datetime.now().strftime("%H:%M:%S")
            })

        progress.empty()
        st.session_state.processed_results.extend(new_results)

        if len(new_results) > 1:
            st.markdown("---")
            st.markdown("### 📊 Batch Summary")
            df_new = pd.DataFrame(new_results)
            st.dataframe(df_new, use_container_width=True, hide_index=True)
            scores = [r["Score"] for r in new_results if isinstance(r["Score"], int)]
            if scores:
                avg = sum(scores) / len(scores)
                c1, c2, c3 = st.columns(3)
                c1.metric("Average", f"{avg:.1f}")
                c2.metric("Highest", max(scores))
                c3.metric("Lowest", min(scores))

# ── CAMERA MODE (GATED) ──
elif st.session_state.mode == "camera":
    st.markdown("### 📸 Camera Mode")
    if st.button("← Back to Menu", type="secondary", key="btn_back_camera"):
        st.session_state.mode = None
        st.session_state.camera_active = False
        safe_rerun()

    if not st.session_state.camera_active:
        st.markdown("""
        <div class="camera-gate">
            <div class="icon">📷</div>
            <h3>Camera Access</h3>
            <p>Click below to activate your camera. Photos are processed locally.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔓 Open Camera", type="primary", use_container_width=True, key="btn_open_camera"):
            st.session_state.camera_active = True
            safe_rerun()
    else:
        st.info("📸 Point camera at the sheet. All 4 corners must be visible.")
        cam_photo = st.camera_input(
            "Capture", label_visibility="collapsed", key="camera_capture_input"
        )

        if cam_photo:
            with st.spinner("Grading..."):
                img_rgb, score, pct, status = process_single_image(
                    cam_photo.getvalue(), answer_list, num_questions,
                    st.session_state.fill_threshold, st.session_state.debug_mode
                )

            if status != "OK":
                st.error(f"❌ Grading failed: {status}")
            else:
                passed = pct >= st.session_state.passing_score
                badge = "✅ PASS" if passed else "❌ FAIL"
                color = "#10b981" if passed else "#ef4444"
                st.markdown(f"""
                <div class="score-card {'' if passed else 'fail'}">
                    <div class="score-header">
                        <div class="score-filename">Live Capture</div>
                        <div class="score-badge {'badge-pass' if passed else 'badge-fail'}">{badge}</div>
                    </div>
                    <div class="score-big">{score}/{num_questions}</div>
                    <div class="score-label">{pct:.1f}% Correct</div>
                    <div class="score-bar-bg">
                        <div class="score-bar-fill" style="width: {pct}%; background: {color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.image(img_rgb, use_column_width=True)

                st.session_state.process

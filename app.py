"""
🏫 Feni Model High School — Professional OMR Grader
Fixed: Answer key moved to main page for mobile visibility.
"""

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import io
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="Feni Model HS OMR Grader",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed"  # Sidebar hidden by default on mobile
)

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
    .answer-box .hint { color: #64748b; font-size: 0.85rem; margin-top: 0.5rem; }
    .mode-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem; }
    @media (max-width: 768px) { .mode-grid { grid-template-columns: 1fr; } }
    .mode-card {
        background: white; border: 2px solid #e2e8f0; border-radius: 16px;
        padding: 1.5rem; text-align: center; cursor: pointer;
        transition: all 0.2s;
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
    div[data-testid="stButton"] > button {
        border-radius: 10px !important; font-weight: 600 !important;
    }
    .instruction-card {
        background: #f8fafc; border-radius: 12px; padding: 1.25rem; margin-top: 1rem;
    }
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
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════
# OMR CONFIGURATION
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
FILL_RATIO_THRESHOLD = 0.35
MULTI_MARK_RELATIVE_MARGIN = 0.28
Y_CLUSTER_TOLERANCE = 18
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
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        edges = cv2.Canny(blurred, 50, 150)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    largest = max(cnts, key=cv2.contourArea)
    img_area = image.shape[0] * image.shape[1]
    if cv2.contourArea(largest) < 0.08 * img_area:
        return None
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
    if len(approx) == 4:
        return approx.reshape(4, 2).astype("float32")
    rect = cv2.minAreaRect(largest)
    return cv2.boxPoints(rect).astype("float32")


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
    return circularity >= MIN_CIRCULARITY


def cluster_bubbles_into_rows(bubbles, y_tol=Y_CLUSTER_TOLERANCE):
    if not bubbles:
        return []
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


def process_column(roi_thresh, roi_color, start_q, answer_key, options, debug=False):
    cnts, _ = cv2.findContours(roi_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    bubbles = [(c, cv2.boundingRect(c)) for c in cnts if is_valid_bubble(c)]
    results = []
    correct = 0
    if len(bubbles) < 12:
        return results, 0
    rows = cluster_bubbles_into_rows(bubbles)
    if len(rows) < 3:
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
        fills = []
        for c, (x, y, w, h) in row[:4]:
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
        if best_val < FILL_RATIO_THRESHOLD:
            marked = None
        elif runner > 0 and ((best_val - runner) / best_val) < MULTI_MARK_RELATIVE_MARGIN:
            marked = "MULTI"
        else:
            marked = options[best_idx]
        correct_letter = answer_key[q_num - 1]
        is_correct = (marked == correct_letter)
        correct += int(is_correct)
        results.append({
            "question": q_num, "marked": marked,
            "correct": correct_letter, "is_correct": is_correct,
        })
        for j, (c, (x, y, w, h)) in enumerate(row[:4]):
            cx, cy = x + w // 2, y + h // 2
            r = max(w, h) // 2 + 1
            letter = options[j]
            if letter == marked and is_correct:
                cv2.circle(roi_color, (cx, cy), r, (0, 255, 0), 3)
            elif letter == marked and not is_correct:
                cv2.circle(roi_color, (cx, cy), r, (0, 0, 255), 3)
            if letter == correct_letter and not is_correct:
                cv2.circle(roi_color, (cx, cy), r, (0, 255, 0), 1)
    return results, correct


def process_single_image(image_bytes, answer_key, num_questions, debug=False):
    pil_img = Image.open(io.BytesIO(image_bytes))
    pil_img = ImageOps.exif_transpose(pil_img)
    pil_img = pil_img.convert("RGB")
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if img is None:
        return None, 0, 0, "Could not read image"
    max_dim = 2000
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
    corners = find_sheet_contour(img)
    if corners is None:
        return None, 0, 0, "Sheet not detected"
    warped = four_point_transform(img, corners, WARPED_SIZE)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    if debug:
        st.sidebar.image(thresh, caption="Threshold", use_column_width=True)
    total_correct = 0
    col_rois = get_absolute_rois(WARPED_SIZE[0], WARPED_SIZE[1])
    for col_idx, (x, y, w, h) in enumerate(col_rois):
        roi_t = thresh[y:y + h, x:x + w]
        roi_c = warped[y:y + h, x:x + w]
        start_q = col_idx * 10 + 1
        if start_q <= num_questions:
            _, cscore = process_column(roi_t, roi_c, start_q, answer_key, OPTIONS, debug)
            total_correct += cscore
    pct = (total_correct / num_questions) * 100
    cv2.putText(warped, f"Score: {total_correct}/{num_questions} ({pct:.1f}%)",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    return warped_rgb, total_correct, pct, "OK"


# ═══════════════════════════════════════════════════════════════
# SIDEBAR — Settings only (Answer key moved to main page)
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("**⚙️ Settings**")
    st.session_state.passing_score = st.slider(
        "Passing %", 0.0, 100.0, 33.0, 1.0, key="sidebar_passing_slider"
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
            st.rerun()

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

# ═══════════════════════════════════════════════════════════════
# STEP 1: ANSWER KEY — PROMINENTLY ON MAIN PAGE
# ═══════════════════════════════════════════════════════════════
st.markdown("### 📝 Step 1: Enter Answer Key")

with st.container():
    st.markdown("""
    <div class="answer-box">
        <h4>✅ Correct Answers</h4>
    </div>
    """, unsafe_allow_html=True)

    default_key = "B," + ",".join(["A"] * 24)
    user_key = st.text_area(
        "Type answers separated by commas (A, B, C, D...)",
        value=default_key,
        height=80,
        key="main_answer_key_input"  # ← Right here on the main page!
    )
    st.markdown('<div class="answer-box hint">💡 <b>A = ক | B = খ | C = গ | D = ঘ</b></div>', unsafe_allow_html=True)

answer_list = [a.strip().upper() for a in user_key.split(",") if a.strip()]
num_questions = len(answer_list)

if num_questions == 0:
    st.warning("⚠️ Please enter at least one answer above.")
    st.stop()
elif num_questions > 30:
    st.error(f"⚠️ Maximum 30 questions supported. You entered {num_questions}.")
    st.stop()

st.success(f"✅ Answer key set: **{num_questions} questions**")

# ═══════════════════════════════════════════════════════════════
# STEP 2: CHOOSE INPUT METHOD
# ═══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(f"### {'📸 Step 2: Choose Input Method' if st.session_state.language == 'English' else '📸 ধাপ ২: ইনপুট পদ্ধতি নির্বাচন করুন'}")

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
            st.rerun()

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
            st.rerun()

    if st.session_state.processed_results:
        st.markdown("---")
        st.markdown("### 📊 Recent Results")
        st.dataframe(pd.DataFrame(st.session_state.processed_results), use_container_width=True, hide_index=True)

# ── UPLOAD MODE ──
elif st.session_state.mode == "upload":
    st.markdown("### 📁 Bulk Upload Mode")
    if st.button("← Back to Menu", type="secondary", key="btn_back_upload"):
        st.session_state.mode = None
        st.rerun()

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
                file.getvalue(), answer_list, num_questions, st.session_state.debug_mode
            )

            if status != "OK":
                st.error(f"❌ **{file.name}**: {status}")
                new_results.append({
                    "Filename": file.name, "Score": "Error", "Percentage": "Error",
                    "Status": status, "Time": datetime.now().strftime("%H:%M:%S")
                })
                continue

            passed = pct >= st.session_state.passing_score
            badge_class = "badge-pass" if passed else "badge-fail"
            badge_text = "PASS" if passed else "FAIL"
            card_class = "" if passed else "fail"
            bar_color = "#10b981" if passed else "#ef4444"

            st.markdown(f"""
            <div class="score-card {card_class}">
                <div class="score-header">
                    <div class="score-filename">{file.name}</div>
                    <div class="score-badge {badge_class}">{badge_text}</div>
                </div>
                <div class="score-big">{score}/{num_questions}</div>
                <div class="score-label">{pct:.1f}% Correct</div>
                <div class="score-bar-bg">
                    <div class="score-bar-fill" style="width: {pct}%; background: {bar_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("🔍 View Annotated Sheet", expanded=False):
                st.image(img_rgb, use_column_width=True)

            new_results.append({
                "Filename": file.name, "Score": score,
                "Percentage": f"{pct:.1f}%",
                "Status": "Pass" if passed else "Fail",
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
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Average", f"{avg:.1f}")
                col_b.metric("Highest", max(scores))
                col_c.metric("Lowest", min(scores))

# ── CAMERA MODE — GATED ──
elif st.session_state.mode == "camera":
    st.markdown("### 📸 Camera Mode")
    if st.button("← Back to Menu", type="secondary", key="btn_back_camera"):
        st.session_state.mode = None
        st.session_state.camera_active = False
        st.rerun()

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
            st.rerun()
    else:
        st.info("📸 Point camera at the sheet. All 4 corners must be visible.")
        cam_photo = st.camera_input(
            "Capture", label_visibility="collapsed", key="camera_capture_input"
        )

        if cam_photo:
            with st.spinner("Grading..."):
                img_rgb, score, pct, status = process_single_image(
                    cam_photo.getvalue(), answer_list, num_questions, st.session_state.debug_mode
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

                st.session_state.processed_results.append({
                    "Filename": f"Camera_{datetime.now().strftime('%H%M%S')}",
                    "Score": score, "Percentage": f"{pct:.1f}%",
                    "Status": "Pass" if passed else "Fail",
                    "Time": datetime.now().strftime("%H:%M:%S")
                })

            if st.button("📸 Take Another Photo", type="primary", use_container_width=True, key="btn_retake_photo"):
                st.rerun()

# ═══════════════════════════════════════════════════════════════
# INSTRUCTIONS
# ═══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(f"### {'📖 Instructions' if st.session_state.language == 'English' else '📖 নিয়মাবলি'}")

if st.session_state.language == "English":
    with st.expander("Click to expand", expanded=False):
        st.markdown("""
        <div class="instruction-card">
        <h4>📸 For Best Results:</h4>
        <ul>
            <li><b>Dark background:</b> Place paper on a dark desk.</li>
            <li><b>All corners visible:</b> Entire sheet must fit in frame.</li>
            <li><b>Good lighting:</b> Avoid shadows on bubbles.</li>
            <li><b>Hold steady:</b> Keep phone parallel to paper.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
else:
    with st.expander("নিয়মাবলি দেখতে ক্লিক করুন", expanded=False):
        st.markdown("""
        <div class="instruction-card">
        <h4>📸 সেরা ফলাফলের জন্য:</h4>
        <ul>
            <li><b>গাঢ় পটভূমি:</b> উত্তরপত্রটি গাঢ় টেবিলের ওপর রাখুন।</li>
            <li><b>৪ কোণ দৃশ্যমান:</b> ছবিতে পুরো কাগজটি উঠে আসতে হবে।</li>
            <li><b>ভালো আলো:</b> ছায়া এড়িয়ে চলুন।</li>
            <li><b>স্থির রাখুন:</b> ফোন কাগজের সমান্তরাল রাখুন।</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="app-footer">
    <p>🏫 Feni Model High School — OMR Grader v2.2</p>
</div>
""", unsafe_allow_html=True)

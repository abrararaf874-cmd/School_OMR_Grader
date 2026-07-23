"""
app.py - Feni Model High School OMR Grader (Robust Mobile Version)
"""

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Feni Model HS OMR Grader",
    page_icon="📝",
    layout="centered"
)

# --- OMR Configuration ---
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
    """Convert relative ROIs to absolute pixel coordinates."""
    return [
        (int(x * width), int(y * height), int(w * width), int(h * height))
        for x, y, w, h in COLUMN_ROIS_REL
    ]


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
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (w, h))


def find_sheet_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Method 1: Otsu thresholding
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Fallback to Canny
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

    # Fallback: rotated bounding box
    rect = cv2.minAreaRect(largest)
    return cv2.boxPoints(rect).astype("float32")


def is_valid_bubble(c):
    x, y, w, h = cv2.boundingRect(c)
    if not (MIN_BUBBLE_DIM <= w <= MAX_BUBBLE_DIM and MIN_BUBBLE_DIM <= h <= MAX_BUBBLE_DIM):
        return False
    if not (0.7 <= w / float(h) <= 1.3):
        return False

    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    if perimeter == 0:
        return False

    circularity = 4 * np.pi * area / (perimeter ** 2)
    return circularity >= MIN_CIRCULARITY


def cluster_bubbles_into_rows(bubbles, y_tolerance=Y_CLUSTER_TOLERANCE):
    if not bubbles:
        return []

    bubbles = sorted(bubbles, key=lambda b: b[1][1])
    rows = []
    current_row = [bubbles[0]]

    for bubble in bubbles[1:]:
        _, (_, y, _, _) = bubble
        row_y_mean = np.mean([b[1][1] for b in current_row])

        if abs(y - row_y_mean) <= y_tolerance:
            current_row.append(bubble)
        else:
            current_row = sorted(current_row, key=lambda b: b[1][0])
            if len(current_row) >= 3:
                rows.append(current_row)
            current_row = [bubble]

    current_row = sorted(current_row, key=lambda b: b[1][0])
    if len(current_row) >= 3:
        rows.append(current_row)

    return rows


def process_column(roi_thresh, roi_color, start_q_num, answer_key, options, debug=False):
    cnts, _ = cv2.findContours(roi_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    bubbles = []
    for c in cnts:
        if is_valid_bubble(c):
            bubbles.append((c, cv2.boundingRect(c)))

    col_results = []
    correct_count = 0

    if len(bubbles) < 12:
        if debug:
            st.warning(f"Col starting Q{start_q_num}: Only {len(bubbles)} bubbles found.")
        return col_results, 0

    rows = cluster_bubbles_into_rows(bubbles)

    if len(rows) < 3:
        if debug:
            st.warning(f"Col starting Q{start_q_num}: Only {len(rows)} rows clustered.")
        return col_results, 0

    for r_idx, row in enumerate(rows):
        q_num = start_q_num + r_idx
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

            bubble_area = cv2.countNonZero(mask)
            if bubble_area == 0:
                fills.append(0.0)
                continue

            filled = cv2.countNonZero(cv2.bitwise_and(roi_thresh, roi_thresh, mask=mask))
            fills.append(filled / float(bubble_area))

        while len(fills) < 4:
            fills.append(0.0)

        best_idx = int(np.argmax(fills))
        best_val = fills[best_idx]
        runner_up = sorted(fills, reverse=True)[1] if len(fills) > 1 else 0.0

        if best_val < FILL_RATIO_THRESHOLD:
            marked = None
        elif runner_up > 0 and ((best_val - runner_up) / best_val) < MULTI_MARK_RELATIVE_MARGIN:
            marked = "MULTI"
        else:
            marked = options[best_idx]

        correct_letter = answer_key[q_num - 1]
        is_correct = (marked == correct_letter)
        correct_count += int(is_correct)

        col_results.append({
            "question": q_num,
            "marked": marked,
            "correct": correct_letter,
            "is_correct": is_correct,
        })

        for j, (c, (x, y, w, h)) in enumerate(row[:4]):
            center = (x + w // 2, y + h // 2)
            radius = max(w, h) // 2 + 1
            letter = options[j]

            if letter == marked and is_correct:
                cv2.circle(roi_color, center, radius, (0, 255, 0), 3)
            elif letter == marked and not is_correct:
                cv2.circle(roi_color, center, radius, (0, 0, 255), 3)
            if letter == correct_letter and not is_correct:
                cv2.circle(roi_color, center, radius, (0, 255, 0), 1)

    return col_results, correct_count


# --- Streamlit UI ---
st.title("🏫 Feni Model High School")
st.subheader("Automated OMR Answer Sheet Grader")

st.sidebar.header("⚙️ Answer Key Options")
st.sidebar.info("Option Key:\n**A = ক | B = খ | C = গ | D = ঘ**")

default_key_str = "B," + ",".join(["A"] * 24)
user_key = st.sidebar.text_area(
    "Enter Answers (Comma-separated, up to 30):",
    value=default_key_str,
    height=120
)
answer_list = [a.strip().upper() for a in user_key.split(",") if a.strip()]

debug_mode = st.sidebar.checkbox("🔍 Debug Mode", value=False)

# Default to Bulk Upload so camera is NEVER requested automatically
input_mode = st.radio(
    "Choose Input Method:",
    ["📁 Bulk Upload (Whole Class)", "📸 Camera (One by One)"],
    index=0,
    horizontal=True
)

uploaded_files = []

if input_mode == "📸 Camera (One by One)":
    # Explicit checkbox prevents browser camera request until user clicks it
    start_cam = st.checkbox("📷 Start Camera", value=False)
    if start_cam:
        cam_photo = st.camera_input("Point camera directly at the sheet")
        if cam_photo:
            uploaded_files.append(cam_photo)
    else:
        st.info("Check **'Start Camera'** above when you are ready to take a picture.")
else:
    uploaded_files = st.file_uploader(
        "Select multiple OMR Photos",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

if uploaded_files:
    num_questions = len(answer_list)

    if num_questions == 0:
        st.error("⚠️ Please enter an answer key in the sidebar.")
    elif num_questions > 30:
        st.error(f"⚠️ Maximum 30 questions supported. You entered {num_questions}.")
    else:
        st.success(f"Grading **{num_questions}** questions. Processing {len(uploaded_files)} paper(s)...")
        class_results = []
        COLUMN_ROIS = get_absolute_rois(WARPED_SIZE[0], WARPED_SIZE[1])

        for file in uploaded_files:
            bytes_data = file.getvalue()

            pil_img = Image.open(io.BytesIO(bytes_data))
            pil_img = ImageOps.exif_transpose(pil_img)
            pil_img = pil_img.convert("RGB")
            image = np.array(pil_img)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if image is None:
                st.error(f"❌ Could not read image **{file.name}**.")
                continue

            max_dim = 2000
            h, w = image.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                image = cv2.resize(image, None, fx=scale, fy=scale)

            corners = find_sheet_contour(image)

            if corners is None:
                st.error(f"❌ Could not detect sheet boundaries in **{file.name}**.")
                if debug_mode:
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image")
                class_results.append({"Filename": file.name, "Score": "Error", "Percentage": "Error"})
                continue

            warped = four_point_transform(image, corners, WARPED_SIZE)
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

            thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            thresh_adaptive = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10
            )
            thresh = thresh_otsu

            if debug_mode:
                c1, c2 = st.columns(2)
                with c1:
                    st.image(thresh_otsu, caption="Otsu Threshold")
                with c2:
                    st.image(thresh_adaptive, caption="Adaptive Threshold")

            total_correct = 0

            for col_idx, (x, y, w, h) in enumerate(COLUMN_ROIS):
                roi_thresh = thresh[y:y + h, x:x + w]
                roi_color = warped[y:y + h, x:x + w]
                start_q = col_idx * 10 + 1

                if start_q <= num_questions:
                    _, col_score = process_column(
                        roi_thresh, roi_color, start_q, answer_list, OPTIONS, debug=debug_mode
                    )
                    total_correct += col_score

                    if debug_mode:
                        st.image(
                            cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB),
                            caption=f"Column {col_idx + 1} (Q{start_q}-Q{start_q + 9})"
                        )

            score_pct = (total_correct / num_questions) * 100
            class_results.append({
                "Filename": file.name,
                "Score": total_correct,
                "Percentage": f"{score_pct:.1f}%"
            })

            cv2.putText(
                warped,
                f"Score: {total_correct}/{num_questions} ({score_pct:.1f}%)",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
            )

            warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
            with st.expander(f"📄 Result for {file.name} — Score: {total_correct}/{num_questions}"):
                st.image(warped_rgb, use_column_width=True)

        if len(class_results) > 1:
            st.subheader("📊 Class Summary Table")
            st.table(class_results)

# --- Instructions ---
st.markdown("---")
col_title, col_lang = st.columns([2.5, 1.5])
with col_lang:
    language = st.selectbox("🌐 Language / ভাষা", ["English", "বাংলা"])
with col_title:
    st.subheader("📖 Instructions" if language == "English" else "📖 ব্যবহারের নিয়ম")

if language == "English":
    with st.expander("Click to view About & How to Use", expanded=False):
        st.markdown("""
### 🏫 About the App
This app is designed for **Feni Model High School** teachers to quickly grade MCQ answer sheets using a mobile camera or uploaded files.

---

### 📱 Step-by-Step Instructions:
1. **Set the Answer Key (Sidebar):** Type correct answers separated by commas (e.g., `A, B, C, D...`).
   * *Key Reference: A = ক | B = খ | C = গ | D = ঘ*
2. **Choose Input Mode:**
   * 📁 **Bulk Upload:** Upload multiple student photos at once.
   * 📸 **Camera:** Check 'Start Camera' to grade sheets one-by-one.
3. **Important Scanning Tips:**
   * Place paper flat on a **dark surface**.
   * Ensure **all 4 corners** are visible.
   * Use good lighting without heavy shadows.
   * Hold phone parallel to the paper (avoid steep angles).
4. **Debug Mode:** Enable in sidebar if grading looks wrong — it shows thresholded images and detected bubbles.
""")
else:
    with st.expander("অ্যাপ পরিচিতি ও নিয়ম দেখতে এখানে ক্লিক করুন", expanded=False):
        st.markdown("""
### 🏫 ওএমআর গ্রেডার পরিচিতি
এই ওয়েব অ্যাপটি **ফেনী মডেল হাই স্কুল**-এর শিক্ষকদের জন্য তৈরি।

---

### 📱 ব্যবহারের নিয়মাবলি:
১. **উত্তরমালা সেটিং (সাইডবার):** কমা দিয়ে সঠিক উত্তরগুলো লিখুন (যেমন: `A, B, C, D...`)।
   * *সংকেত: A = ক | B = খ | C = গ | D = ঘ*
২. **মোড নির্বাচন করুন:**
   * 📁 **বাল্ক আপলোড:** একসাথে পুরো ক্লাসের ছবি আপলোড করুন!
   * 📸 **ক্যামেরা:** 'Start Camera' চেক বক্সে টিক দিয়ে একে একে প্রতিটি উত্তরপত্রের ছবি তুলুন।
৩. **ছবি তোলার জরুরি টিপস:**
   * কাগজটি একটি অন্ধকার বা গাঢ় টেবিলের ওপর সোজা করে রাখুন।
   * ছবির ভেতরে যেন উত্তরপত্রের **৪টি কোণই** পরিষ্কার দেখা যায়।
   * পর্যাপ্ত আলো বজায় রাখুন যেন কোনো ছায়া না পড়ে।
৪. **ফলাফল দেখুন:** অ্যাপটি সাথে সাথে মোট নম্বর হিসাব করবে, ছবিতে সঠিক (সবুজ) ও ভুল (লাল) চিহ্নিত করবে এবং পুরো ক্লাসের ফলাফল তালিকা দেখাবে!
""")

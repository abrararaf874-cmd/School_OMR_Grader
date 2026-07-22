"""
app.py - Feni Model High School OMR Grader (Dynamic Question Count)
"""

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="Feni Model HS OMR Grader",
    page_icon="📝",
    layout="centered"
)

# --- OMR Configuration ---
OPTIONS = ["A", "B", "C", "D"]
WARPED_SIZE = (850, 1100)

MIN_BUBBLE_DIM = 12
MAX_BUBBLE_DIM = 40
FILL_RATIO_THRESHOLD = 0.40
MULTI_MARK_MARGIN = 0.12

COLUMN_ROIS = [
    (340, 580, 160, 480),
    (500, 580, 160, 480),
    (660, 580, 160, 480)
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
    edged = cv2.Canny(blurred, 50, 150)
    edged = cv2.dilate(edged, None, iterations=1)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    img_area = image.shape[0] * image.shape[1]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.25 * img_area:
            return approx.reshape(4, 2).astype("float32")
    return None

def sort_contours_spatial(cnts):
    boxes = [cv2.boundingRect(c) for c in cnts]
    sorted_pairs = sorted(zip(cnts, boxes), key=lambda b: b[1][1])
    rows = []
    for i in range(0, len(sorted_pairs), 4):
        group = sorted_pairs[i:i+4]
        group = sorted(group, key=lambda b: b[1][0])
        rows.append([c for c, box in group])
    return rows

def process_column(roi_thresh, roi_color, start_q_num, answer_key, options):
    cnts, _ = cv2.findContours(roi_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    bubbles = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if MIN_BUBBLE_DIM <= w <= MAX_BUBBLE_DIM and MIN_BUBBLE_DIM <= h <= MAX_BUBBLE_DIM:
            if 0.75 <= w / float(h) <= 1.25:
                bubbles.append(c)

    col_results = []
    correct_count = 0

    # Ensure bubbles are found and are in multiples of 4 (A, B, C, D)
    if len(bubbles) == 0 or len(bubbles) % 4 != 0:
        return col_results, 0

    rows = sort_contours_spatial(bubbles)

    for r_idx, row in enumerate(rows):
        q_num = start_q_num + r_idx
        
        # STOP processing if we have reached the end of the teacher's answer key
        if q_num > len(answer_key):
            break
            
        fills = []

        for c in row:
            mask = np.zeros(roi_thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            bubble_area = cv2.countNonZero(mask)
            filled = cv2.countNonZero(cv2.bitwise_and(roi_thresh, roi_thresh, mask=mask))
            fills.append(filled / float(bubble_area) if bubble_area else 0.0)

        best_idx = int(np.argmax(fills))
        best_val = fills[best_idx]
        runner_up = sorted(fills, reverse=True)[1] if len(fills) > 1 else 0.0

        if best_val < FILL_RATIO_THRESHOLD:
            marked = None
        elif (best_val - runner_up) < MULTI_MARK_MARGIN:
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
            "is_correct": is_correct
        })

        for j, c in enumerate(row):
            x, y, w, h = cv2.boundingRect(c)
            center = (x + w // 2, y + h // 2)
            radius = max(w, h) // 2 + 1
            letter = options[j]

            if letter == marked and is_correct:
                cv2.circle(roi_color, center, radius, (0, 200, 0), 2)
            elif letter == marked and not is_correct:
                cv2.circle(roi_color, center, radius, (0, 0, 255), 2)
            if letter == correct_letter and not is_correct:
                cv2.circle(roi_color, center, radius, (0, 200, 0), 1)

    return col_results, correct_count


# --- Streamlit UI ---
st.title("🏫 Feni Model High School")
st.subheader("Automated OMR Answer Sheet Grader")
st.write("Grade a single sheet or upload a whole class batch at once!")

st.sidebar.header("⚙️ Answer Key Options")
st.sidebar.info("Option Key:\n**A = ক | B = খ | C = গ | D = ঘ**")

# Defaulting to 25 answers to show it's dynamic
default_key_str = "B," + ",".join(["A"] * 24) 
user_key = st.sidebar.text_area(
    "Enter Answers (Comma-separated, up to 30):",
    value=default_key_str,
    height=120
)

answer_list = [a.strip().upper() for a in user_key.split(",") if a.strip()]

input_mode = st.radio("Choose Input Method:", ["📸 Camera (One by One)", "📁 Bulk Upload (Whole Class)"], horizontal=True)

uploaded_files = []
if input_mode == "📸 Camera (One by One)":
    cam_photo = st.camera_input("Point camera directly at the sheet")
    if cam_photo:
        uploaded_files.append(cam_photo)
else:
    uploaded_files = st.file_uploader("Select multiple OMR Photos from your gallery", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    num_questions = len(answer_list)
    
    if num_questions == 0:
        st.error("⚠️ Please enter an answer key in the sidebar.")
    elif num_questions > 30:
        st.error(f"⚠️ Maximum 30 questions supported. You entered {num_questions}.")
    else:
        st.success(f"Grading out of **{num_questions} questions**. Processing {len(uploaded_files)} paper(s)...")
        
        class_results = []

        for file in uploaded_files:
            bytes_data = file.getvalue()
            file_bytes = np.frombuffer(bytes_data, np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            corners = find_sheet_contour(image)

            if corners is None:
                st.error(f"❌ Could not detect sheet boundaries in **{file.name}**.")
                class_results.append({"Filename": file.name, "Score": "Error", "Percentage": "Error"})
            else:
                warped = four_point_transform(image, corners, WARPED_SIZE)
                gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

                total_correct = 0
                for col_idx, (x, y, w, h) in enumerate(COLUMN_ROIS):
                    roi_thresh = thresh[y:y+h, x:x+w]
                    roi_color = warped[y:y+h, x:x+w]
                    start_q = col_idx * 10 + 1
                    
                    # Only process the column if we haven't exceeded the total questions
                    if start_q <= num_questions:
                        _, col_score = process_column(roi_thresh, roi_color, start_q, answer_list, OPTIONS)
                        total_correct += col_score

                score_pct = (total_correct / num_questions) * 100
                class_results.append({"Filename": file.name, "Score": total_correct, "Percentage": f"{score_pct:.1f}%"})

                cv2.putText(warped, f"Score: {total_correct}/{num_questions} ({score_pct:.1f}%)",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

                with st.expander(f"📄 Result for {file.name} - Score: {total_correct}/{num_questions}"):
                    st.image(warped_rgb, use_column_width=True)

        if len(class_results) > 1:
            st.subheader("📊 Class Summary Table")
            st.table(class_results)

# --- Instructions & Language Selector ---
st.markdown("---")

# Place Instruction header and Language dropdown side-by-side
col_title, col_lang = st.columns([2.5, 1.5])

with col_lang:
    language = st.selectbox("🌐 Language / ভাষা", ["English", "বাংলা"])

with col_title:
    st.subheader("📖 Instructions" if language == "English" else "📖 ব্যবহারের নিয়ম")

if language == "English":
    with st.expander("Click to view About & How to Use", expanded=False):
        st.markdown("""
        ### 🏫 About the App
        This app is designed for **Feni Model High School** teachers to quickly and accurately grade multiple-choice (MCQ) answer sheets using a mobile camera or uploaded files.

        ---

        ### 📱 Step-by-Step Instructions:
        1. **Set the Answer Key (Sidebar):** Type your correct answers separated by commas (e.g., `A, B, C, D...`). 
           * *Key Reference: A = ক | B = খ | C = গ | D = ঘ*
        2. **Choose Input Mode:**
           * 📸 **Camera:** Grade sheets live one-by-one.
           * 📁 **Bulk Upload:** Upload up to 120+ student photos from your gallery at once!
        3. **Important Scanning Tips:**
           * Place paper flat on a dark surface (like a desk).
           * Ensure **all 4 corners** of the sheet are visible in the photo.
           * Make sure the room is well-lit without heavy shadows.
        4. **View Results:** See instant total scores, annotated green/red marked sheets, and a complete class summary table!
        """)
else:
    with st.expander("অ্যাপ পরিচিতি ও নিয়ম দেখতে এখানে ক্লিক করুন", expanded=False):
        st.markdown("""
        ### 🏫 ওএমআর গ্রেডার পরিচিতি
        এই ওয়েব অ্যাপটি **ফেনী মডেল হাই স্কুল**-এর শিক্ষকদের জন্য তৈরি করা হয়েছে। এর মাধ্যমে যেকোনো স্মার্টফোন ক্যামেরা বা কম্পিউটারের সাহায্যে খুব সহজেই এবং দ্রুত বহুনির্বাচনী (MCQ) উত্তরপত্র মূল্যায়ন করা যাবে।

        ---

        ### 📱 ব্যবহারের নিয়মাবলি:
        ১. **উত্তরমালা সেটিং (সাইডবার):** সাইডবারে কমা দিয়ে সঠিক উত্তরগুলো লিখুন (যেমন: `A, B, C, D...`)। 
           * *সংকেত: A = ক | B = খ | C = গ | D = ঘ*
        ২. **মোড নির্বাচন করুন:**
           * 📸 **ক্যামেরা:** একে একে প্রতিটি উত্তরপত্রের ছবি তুলে চেক করুন।
           * 📁 **বাল্ক আপলোড:** একসাথে পুরো ক্লাসের (১২০+ টি) ছবি গ্যালারি থেকে সিলেক্ট করে আপলোড করুন!
        ৩. **ছবি তোলার জরুরি টিপস:**
           * কাগজটি একটি অন্ধকার বা গাঢ় টেবিলের ওপর সোজা করে রাখুন।
           * ছবির ভেতরে যেন উত্তরপত্রের **৪টি কোণই** পরিষ্কার দেখা যায়।
           * পর্যাপ্ত আলো বজায় রাখুন যেন কোনো ছায়া না পড়ে।
        ৪. **ফলাফল দেখুন:** অ্যাপটি সাথে সাথে মোট নম্বর হিসাব করবে, ছবিতে সঠিক (সবুজ) ও ভুল (লাল) চিহ্নিত করবে এবং পুরো ক্লাসের ফলাফল তালিকা দেখাবে!
        """)

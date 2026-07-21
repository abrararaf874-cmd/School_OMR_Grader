# 🏫 Feni Model High School - Automated OMR Grader

Welcome to the Automated OMR (Optical Mark Recognition) Grader! This web application is designed specifically for teachers to instantly grade 30-question Multiple Choice Question (MCQ) answer sheets using a smartphone camera or computer.

Built with Python, OpenCV, and Streamlit, this tool eliminates the need for manual grading, saving time and reducing human error.

## ✨ Features
* **No Installation Required:** Runs entirely in your mobile or desktop web browser.
* **Live Camera Integration:** Snap a photo of the answer sheet directly from your phone.
* **Custom Answer Keys:** Easily input the correct answers for any test.
* **Visual Feedback:** Generates an annotated image highlighting correct (green) and incorrect (red) marks.
* **Detailed Breakdown:** Provides a question-by-question table showing what the student marked versus the correct answer.

---

## 📱 How to Use the App

### Step 1: Open the App
Click the provided Streamlit web link on your smartphone or computer browser. 

### Step 2: Enter the Answer Key
1. Tap the double-arrow icon (`>>`) in the top-left corner to open the sidebar.
2. In the text box, enter the 30 correct answers separated by commas (e.g., `A, B, C, D, A...`).
   * *Note: A = ক, B = খ, C = গ, D = ঘ*

### Step 3: Scan the Answer Sheet
1. Choose your input method: **Take Photo with Camera** or **Upload Image**.
2. If using the camera, hold your phone directly above the paper.
3. **Important Scanning Tips for Best Results:**
   * Ensure the paper is on a flat, contrasting surface (like a dark desk).
   * Make sure **all 4 corners** of the paper are fully visible inside the camera frame.
   * Ensure the room is well-lit so there are no heavy shadows across the bubbles.

### Step 4: View Results
Once you upload or take the photo, the app will process the image for a few seconds. It will automatically output:
* The final numerical score (e.g., 25/30) and percentage.
* A marked-up photo showing exactly which bubbles were correct or wrong.
* A detailed summary table at the bottom of the page.

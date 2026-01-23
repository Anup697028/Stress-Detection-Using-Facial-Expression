# Autism Emotion Detection Full-Stack Web Application

This project provides a full-stack web application to detect and analyze facial emotions and predict autism likelihood. It offers a comprehensive system for assisting medical professionals by providing objective data and visual evidence.

## Features

*   **User Authentication:** Secure login and registration system for users.
*   **Informative Home Page:** Provides an overview of the project and its real-world applications.
*   **Professional UI/UX:** Clean, intuitive, and medical-themed design with subtle background animations for an enhanced user experience.
*   **Integrated Navigation Bar:** The home page features a fixed navigation bar with direct links for Login and Register.
*   **Webcam Integration:** Captures live video feed from the user's webcam for real-time emotion analysis.
*   **Single Image Prediction:** Predicts autism likelihood from a single uploaded image.
*   **Video Emotion Analysis:** Analyzes emotions from an uploaded video file.
*   **Real-time Emotion Display:** During webcam analysis, detected faces are outlined with bounding boxes and their real-time emotion labels are displayed.
*   **Continuous Emotion Detection:** Analyzes facial expressions for emotions (Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised) continuously over a 15-second window.
*   **Autism Suspicion Flagging:** Implements decision rules to flag a person as "Stress Suspected" if their *valid* emotions (excluding "N/A") fluctuate across 3 or more distinct categories within 15 seconds. Otherwise, it flags as "Not Autism."
*   **Video Evidence (Stress Suspected):** For "Stress Suspected" cases, the entire 15-second video recording (from webcam or uploaded video) is saved.
*   **Patient Data Storage:** Stores analysis results (Patient Name, Date & Time, Detected Status, Emotion Sequence, Link to Video Proof) in a JSON-based database (`autism_results.json`).
*   **Admin Results Page:** A dedicated page where **the administrator can view all recorded patient analysis results, including patient details, analysis outcomes, and the ability to download video proofs for individuals flagged as "Stress Suspected."**

## Technologies Used

### Backend (Python, Flask)

*   **Python (Version 3.8.20):** Programming language.
*   **Flask:** Web framework for building the API and serving the frontend.
*   **SQLite:** Lightweight database for user management.
*   **TensorFlow/Keras:** For loading and running the pre-trained emotion detection (`emotion_model.h5`) and autism prediction (`model.h5`) models.
*   **OpenCV (cv2):** For image processing, face detection, and video handling.
*   **`flask-cors`:** For handling Cross-Origin Resource Sharing.
*   **Pillow (PIL):** For image manipulation (used in single image prediction).
*   **`werkzeug.security`:** For password hashing and checking.
*   **`werkzeug.utils`:** For secure filename handling during file uploads.

### Frontend (HTML, CSS, JavaScript)

*   **HTML5:** Structure of the web pages (`backend/templates/home.html`, `login.html`, `register.html`, `index.html`, `admin_results.html`).
*   **CSS3:** Styling of the web pages, including a professional, medical-themed look and subtle background animations (`backend/static/style.css`).
*   **JavaScript:** Client-side logic for webcam access, sending data to the backend, displaying real-time feedback, and managing results (`backend/static/script.js`).

## Setup and Running Instructions

Follow these steps to set up and run the application locally.

### 1. Project Setup

1.  **Navigate to the backend directory:**
    ```bash
    cd C:\Users\YOGESH\OneDrive\Desktop\autism
    ```

2.  **Install Python Dependencies:**
    It is highly recommended to use a virtual environment.

    ```bash
    # Create a virtual environment
    python -m venv venv

    # Activate the virtual environment
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate

    # Install dependencies
    pip install -r requirements.txt
    ```

3.  **Ensure Model and Cascade Files Are Present:**
    Make sure the `emotion_model.h5`, `model.h5`, `haarcascade_frontalface_default.xml`, and `labels.txt` files are located in the `autism/` directory.

### 2. Run the Application

1.  **Start the Flask Backend:**
    Make sure you are in the `autism` directory (as per step 1 of Project Setup) and your virtual environment is activated.
    ```bash
    python app.py
    ```
    The Flask application will start running, usually on `http://127.0.0.1:5000`.

2.  **Access the Frontend:**
    Open your web browser and navigate to `http://127.0.0.1:5000/`.
    You will land on the Home page with a navigation bar where you can find Login and Register options.

## Usage

Once the application is running and you have accessed the frontend in your browser:

1.  **Login or Register:**
    *   If you are a new user, navigate to the `/register` page and create an account.
    *   If you have an account, navigate to the `/login` page and log in.
    *   After successful login, you will be redirected to the main detection page (`/`).
    **Default Admin Login:** For administrative access to view all patient results and download videos, use the following credentials:
    *   **Username:** `admin`
    *   **Password:** `123`

2.  **Enter Patient Name:** Fill in the "Patient Name" field on the detection page.

3.  **Webcam Analysis:**
    *   Click the "**Start Analysis**" button to begin real-time emotion detection from your webcam.
    *   Grant webcam access if prompted by your browser.
    *   During the 15-second recording, you will see a live video feed with detected faces outlined by **bounding boxes** and their **real-time emotion labels** displayed on the screen.
    *   After 15 seconds, the application will process emotions, and display the overall autism suspicion status and the sequence of detected emotions. A video proof link will appear if autism is suspected.

4.  **Image Upload for Autism Prediction:**
    *   Click "Choose File" under "Upload Image for Autism Prediction" to select an image file.
    *   Click "Upload Image" to send the image to the backend for autism prediction. Results (Class and Confidence) will be displayed below.

5.  **Video Upload for Emotion Analysis:**
    *   Click "Choose File" under "Upload Video for Emotion Analysis" to select a video file.
    *   Click "Upload Video" to send the video to the backend for emotion analysis. The results will be displayed similar to the webcam analysis, including a video proof link if autism is suspected.

6.  **Access Admin Results:**
    *   On the detection page, click the "View All Results" button to navigate to the admin page. Here you can see a table of all patient analysis results, including downloadable video proofs.

7.  **Logout:**
    *   Click the "Logout" button on the detection page to end your session and return to the home page.

## Output

*   **Video Files:** If a patient is flagged as "Stress Suspected" (from webcam or video upload), a video file will be saved in the `backend/output_videos/` directory. The video will be named with the patient's name, analysis type (e.g., `AutismCheck` or `AutismCheck_VideoUpload`), and timestamp.
*   **Results Database:** All analysis results (from webcam, image, and video analysis) are stored in `backend/autism_results.json`.
*   **User Database:** User credentials (hashed passwords) are stored in `backend/database.db`.

## Important Notes and Troubleshooting

*   **Webcam Access:** Browsers generally require HTTPS for webcam access in production. For local development, `http://localhost` or `http://127.0.0.1` usually allows webcam access without HTTPS. If you face issues, check your browser's security settings.
*   **Model Loading:** Ensure all model files (`emotion_model.h5`, `model.h5`, `haarcascade_frontalface_default.xml`, `labels.txt`) are in the correct `backend` directory. Errors during model loading will cause the backend to fail.
*   **Port Conflicts:** If port 5000 is already in use, the Flask application will fail to start. You can change the port in `backend/app.py` if needed.
*   **Dependencies:** Double-check that all dependencies listed in `backend/requirements.txt` are correctly installed in your virtual environment. If you encounter `ModuleNotFoundError`, you likely need to install missing packages.
*   **Large Files:** Uploading very large video files might take considerable time and consume significant memory. For production, consider implementing streaming or more robust file handling.

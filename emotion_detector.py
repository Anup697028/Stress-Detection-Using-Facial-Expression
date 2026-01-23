import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import time
import datetime
import os
import json

# Model architecture (must match the saved model's architecture)
def build_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    return model

# Load the trained weights and build the model
model = build_model()
model.load_weights('emotion_model.h5')

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load face cascade classifier
facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

output_video_dir = "output_videos"
if not os.path.exists(output_video_dir):
    os.makedirs(output_video_dir)

results_db_path = "autism_results.json"

def analyze_emotions_from_frames(frames, patient_name, normal_stress_threshold=30.0, high_stress_threshold=60.0):
    emotion_sequence = []
    annotated_frames_to_save = [] # New list to store frames with annotations drawn

    for frame_data in frames:
        frame = frame_data.copy() # Work on a copy to draw annotations

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        detected_emotion = "N/A"
        if len(faces) > 0:
            # Process only the largest face for simplicity
            (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            detected_emotion = emotion_dict[maxindex]

            # Draw bounding box and text on the frame copy
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 2) # Green bounding box
            cv2.putText(frame, detected_emotion, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        emotion_sequence.append(detected_emotion)
        annotated_frames_to_save.append(frame) # Add the annotated frame to the list

    # Analyze emotion sequence
    stress_suspected = False
    # Filter out "N/A" emotions before counting unique categories
    filtered_emotions = [e for e in emotion_sequence if e != "N/A"]
    unique_emotions = set(filtered_emotions)
    num_unique_emotions = len(unique_emotions)

    # Decision Rules:
    # If the number of unique emotions is GREATER THAN 3, then flag as "Stress Suspected."
    if num_unique_emotions > 3:
        stress_suspected = True
    # Otherwise (3 or fewer unique emotions), flag as "Not Autism."
    else:
        stress_suspected = False

    # --- New Emotion Analysis for Stress Level --- START
    stress_emotions = ["Fearful", "Surprised", "Sad"]
    happy_emotions = ["Happy"]
    neutral_emotions = ["Neutral"]

    total_relevant_emotions_count = 0
    stress_emotion_count = 0
    happy_emotion_count = 0
    neutral_emotion_count = 0

    for emotion in filtered_emotions:
        if emotion in stress_emotions:
            stress_emotion_count += 1
            total_relevant_emotions_count += 1
        elif emotion in happy_emotions:
            happy_emotion_count += 1
            total_relevant_emotions_count += 1
        elif emotion in neutral_emotions:
            neutral_emotion_count += 1
            total_relevant_emotions_count += 1

    stress_percentage = 0.0
    if total_relevant_emotions_count > 0:
        stress_percentage = (stress_emotion_count / total_relevant_emotions_count) * 100

    # Define thresholds
    # NORMAL_STRESS_THRESHOLD = 30.0 # Configurable variable
    # HIGH_STRESS_THRESHOLD = 60.0 # Configurable variable
    # Use passed thresholds
    NORMAL_STRESS_THRESHOLD = normal_stress_threshold
    HIGH_STRESS_THRESHOLD = high_stress_threshold

    stress_level_status = "Undefined"

    is_happy_predominant = False
    is_neutral_predominant = False

    if happy_emotion_count > stress_emotion_count and happy_emotion_count > neutral_emotion_count:
        is_happy_predominant = True
        stress_level_status = "No Stress (Happy)"
    elif neutral_emotion_count > stress_emotion_count and neutral_emotion_count > happy_emotion_count:
        is_neutral_predominant = True
        stress_level_status = "No Stress (Neutral)"
    elif stress_percentage > HIGH_STRESS_THRESHOLD:
        stress_level_status = "High Stress"
    elif stress_percentage > NORMAL_STRESS_THRESHOLD:
        stress_level_status = "Normal Stress"
    else:
        stress_level_status = "No Stress (Low Percentage)"

    # --- New Emotion Analysis for Stress Level --- END

    # Prepare data for saving
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    video_link = "N/A"
    if stress_suspected or stress_level_status in ["Normal Stress", "High Stress"]:
        status = "Stress Suspected"
        # Save video (use annotated_frames_to_save)
        if annotated_frames_to_save:
            frame_height, frame_width, _ = annotated_frames_to_save[0].shape
            video_filename = f"{patient_name}_AutismCheck_{current_datetime}.mp4"
            video_filepath = os.path.join(output_video_dir, video_filename)
            fourcc = cv2.VideoWriter_fourcc(*'H264') # Changed codec to H264
            out = cv2.VideoWriter(video_filepath, fourcc, 20.0, (frame_width, frame_height))

            for frame in annotated_frames_to_save:
                out.write(frame)
            out.release()
            video_link = video_filepath
    else:
        status = "Stress not detected"

    # Store results in a simple JSON database
    result_entry = {
        "Patient Name": patient_name,
        "Date & Time": current_datetime,
        "Detected Status": status,
        "Emotion Sequence": emotion_sequence,
        "Stress Percentage": round(stress_percentage, 2), # New: Store stress percentage
        "Stress Level Status": stress_level_status,      # New: Store stress level status
        "Link to saved video proof": video_link
    }

    try:
        with open(results_db_path, 'r+') as f:
            data = json.load(f)
            data.append(result_entry)
            f.seek(0)
            json.dump(data, f, indent=4)
    except FileNotFoundError:
        with open(results_db_path, 'w') as f:
            json.dump([result_entry], f, indent=4)

    return {
        "status": status,
        "emotion_sequence": emotion_sequence,
        "stress_percentage": round(stress_percentage, 2), # New: return stress percentage
        "stress_level_status": stress_level_status,       # New: return stress level status
        "video_link": video_link
    }

def get_emotion_for_single_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    detected_emotion = "N/A"
    bounding_box = None # (x, y, w, h)

    if len(faces) > 0:
        # Process only the largest face for simplicity
        (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        detected_emotion = emotion_dict[maxindex]
        bounding_box = (int(x), int(y), int(w), int(h))

    return {"emotion": detected_emotion, "bounding_box": bounding_box}

def analyze_emotions_from_video(video_path, patient_name, normal_stress_threshold=30.0, high_stress_threshold=60.0):
    emotion_sequence = []
    annotated_frames_to_save = [] # New list to store annotated frames for saving
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {"status": "Error", "message": "Could not open video file."}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame_copy = frame.copy() # Work on a copy to draw annotations

        gray = cv2.cvtColor(current_frame_copy, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        detected_emotion = "N/A"
        if len(faces) > 0:
            (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            detected_emotion = emotion_dict[maxindex]

            # Draw bounding box and text on the frame copy
            cv2.rectangle(current_frame_copy, (x, y-50), (x+w, y+h+10), (0, 255, 0), 2) # Green bounding box
            cv2.putText(current_frame_copy, detected_emotion, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        emotion_sequence.append(detected_emotion)
        annotated_frames_to_save.append(current_frame_copy) # Add the annotated frame

    cap.release()

    # Analyze emotion sequence (same logic as analyze_emotions_from_frames)
    filtered_emotions = [e for e in emotion_sequence if e != "N/A"]
    unique_emotions = set(filtered_emotions)
    num_unique_emotions = len(unique_emotions)

    stress_suspected = False
    if num_unique_emotions > 3:
        stress_suspected = True
    else:
        stress_suspected = False

    # --- New Emotion Analysis for Stress Level for Video --- START
    stress_emotions = ["Fearful", "Surprised", "Sad"]
    happy_emotions = ["Happy"]
    neutral_emotions = ["Neutral"]

    total_relevant_emotions_count = 0
    stress_emotion_count = 0
    happy_emotion_count = 0
    neutral_emotion_count = 0

    for emotion in filtered_emotions:
        if emotion in stress_emotions:
            stress_emotion_count += 1
            total_relevant_emotions_count += 1
        elif emotion in happy_emotions:
            happy_emotion_count += 1
            total_relevant_emotions_count += 1
        elif emotion in neutral_emotions:
            neutral_emotion_count += 1
            total_relevant_emotions_count += 1

    stress_percentage = 0.0
    if total_relevant_emotions_count > 0:
        stress_percentage = (stress_emotion_count / total_relevant_emotions_count) * 100

    # Define thresholds (these should ideally be globally configurable)
    # NORMAL_STRESS_THRESHOLD = 30.0
    # HIGH_STRESS_THRESHOLD = 60.0
    # Use passed thresholds
    NORMAL_STRESS_THRESHOLD = normal_stress_threshold
    HIGH_STRESS_THRESHOLD = high_stress_threshold

    stress_level_status = "Undefined"

    is_happy_predominant = False
    is_neutral_predominant = False

    if happy_emotion_count > stress_emotion_count and happy_emotion_count > neutral_emotion_count:
        is_happy_predominant = True
        stress_level_status = "No Stress (Happy)"
    elif neutral_emotion_count > stress_emotion_count and neutral_emotion_count > happy_emotion_count:
        is_neutral_predominant = True
        stress_level_status = "No Stress (Neutral)"
    elif stress_percentage > HIGH_STRESS_THRESHOLD:
        stress_level_status = "High Stress"
    elif stress_percentage > NORMAL_STRESS_THRESHOLD:
        stress_level_status = "Normal Stress"
    else:
        stress_level_status = "No Stress (Low Percentage)"
    # --- New Emotion Analysis for Stress Level for Video --- END

    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    video_link = "N/A"
    if stress_suspected or stress_level_status in ["Normal Stress", "High Stress"]:
        status = "Stress Suspected"
        if annotated_frames_to_save:
            frame_height, frame_width, _ = annotated_frames_to_save[0].shape
            video_filename = f"{patient_name}_StressCheck_VideoUpload_{current_datetime}.mp4"
            video_filepath = os.path.join(output_video_dir, video_filename)
            fourcc = cv2.VideoWriter_fourcc(*'H264') # Changed codec to H264
            out = cv2.VideoWriter(video_filepath, fourcc, 20.0, (frame_width, frame_height))
            for annotated_frame in annotated_frames_to_save:
                out.write(annotated_frame)
            out.release()
            video_link = video_filepath
    else:
        status = "Stress not detected"

    result_entry = {
        "Patient Name": patient_name,
        "Date & Time": current_datetime,
        "Detected Status": status,
        "Emotion Sequence": emotion_sequence,
        "Stress Percentage": round(stress_percentage, 2), # New: Store stress percentage
        "Stress Level Status": stress_level_status,      # New: Store stress level status
        "Link to saved video proof": video_link
    }

    try:
        with open(results_db_path, 'r+') as f:
            data = json.load(f)
            data.append(result_entry)
            f.seek(0)
            json.dump(data, f, indent=4)
    except FileNotFoundError:
        with open(results_db_path, 'w') as f:
            json.dump([result_entry], f, indent=4)

    return {
        "status": status,
        "emotion_sequence": emotion_sequence,
        "stress_percentage": round(stress_percentage, 2), # New: return stress percentage
        "stress_level_status": stress_level_status,       # New: return stress level status
        "video_link": video_link
    }

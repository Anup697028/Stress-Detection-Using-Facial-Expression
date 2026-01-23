from flask import Flask, request, jsonify, render_template, send_from_directory, session, redirect, url_for, flash
import os # For os.path.basename
from functools import wraps # Added for login_required decorator
from flask_cors import CORS
import base64
import numpy as np
import cv2
from PIL import Image, ImageOps
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
from reportlab.pdfgen import canvas
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import tempfile # Added for temporary file creation
from dotenv import load_dotenv # Import load_dotenv
from reportlab.lib.colors import blue # Added for blue color in header
import google.generativeai as genai # Import Google Generative AI library
from collections import Counter # Added for emotion sequence summarization

from emotion_detector import analyze_emotions_from_frames, analyze_emotions_from_video, get_emotion_for_single_frame
from database import init_db, add_user, get_user, get_all_analysis_results

load_dotenv() # Load environment variables from .env

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your_super_secret_key_here') # Get from .env
CORS(app) # Enable CORS for all routes

# Email configuration (retrieved from .env)
sender_email = os.environ.get('EMAIL_SENDER')
sender_password = os.environ.get('EMAIL_PASSWORD')
smtp_server = os.environ.get('EMAIL_SMTP_SERVER')
smtp_port = int(os.environ.get('EMAIL_SMTP_PORT', 587))

# GenAI API Key (retrieved from .env)
GENAI_API_KEY = os.environ.get('GENAI_API_KEY')

genai_model = None
if not GENAI_API_KEY:
    print("Warning: GENAI_API_KEY not found in .env. GenAI suggestions will not be available.")
    genai_model = None
else:
    genai.configure(api_key=GENAI_API_KEY)
    genai_model = genai.GenerativeModel("gemini-2.0-flash")  

# Stress Thresholds (configurable via .env)
NORMAL_STRESS_THRESHOLD = float(os.environ.get('NORMAL_STRESS_THRESHOLD', 30.0))
HIGH_STRESS_THRESHOLD = float(os.environ.get('HIGH_STRESS_THRESHOLD', 60.0))

# Initialize database
with app.app_context():
    init_db()

# Configure directories
VIDEO_SAVE_FOLDER = os.path.join(os.path.dirname(__file__), 'output_videos')
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
REPORTS_FOLDER = os.path.join(os.path.dirname(__file__), 'reports') # New folder for reports

if not os.path.exists(VIDEO_SAVE_FOLDER):
    os.makedirs(VIDEO_SAVE_FOLDER)
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(REPORTS_FOLDER):
    os.makedirs(REPORTS_FOLDER)

# Load the single image stress detection model and labels
AUTISM_MODEL = None
AUTISM_CLASS_NAMES = None
try:
    from keras.models import load_model
    AUTISM_MODEL = load_model("model.h5", compile=False)
    with open("labels.txt", "r") as f:
        AUTISM_CLASS_NAMES = f.readlines()
except Exception as e:
    print(f"Error loading stress detection model or labels: {e}")
    AUTISM_MODEL = None
    AUTISM_CLASS_NAMES = None

def allowed_video_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

# Decorator for login required
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session or session.get('role') != 'admin':
            # Optionally, redirect to a different page or show an access denied message
            flash('You do not have administrative access.', 'error')
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function

# Header and Footer - Moved to module level
def header_footer(canvas, doc, **kwargs):
    canvas.saveState()
    page_width, page_height = doc.pagesize # Access pagesize from doc
    # Header
    canvas.setFillColor(blue) # Set header text color to blue
    canvas.setFont('Helvetica-Bold', 9)
    canvas.drawString(inch, page_height - 0.75 * inch, "Stress Detector") # Changed project name
    canvas.setFont('Helvetica', 9)
    canvas.setFillColorRGB(0,0,0) # Reset color to black for other elements if needed
    canvas.drawString(page_width - inch - 100, page_height - 0.75 * inch, f"Date: {datetime.date.today().strftime('%Y-%m-%d')}")

    # Footer
    canvas.setFont('Helvetica', 9)
    canvas.drawCentredString(page_width / 2.0, 0.75 * inch, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()

def get_genai_suggestions(stress_level):
    """
    Generates personalized medical suggestions based on the stress level using Google's Gemini AI.
    Returns structured text with headings and bullet points.
    """
    if not genai_model:
        # Fallback to hardcoded suggestions if model is not loaded (e.g., API key missing)
        print("GenAI model not available. Returning hardcoded suggestions.")
        return [
            "<b>Precautions and Lifestyle Changes:</b>",
            "- Consult a healthcare professional for a comprehensive evaluation.",
            "- Maintain a balanced diet and regular exercise routine.",
            "- Ensure adequate sleep (7-9 hours per night for adults).",
            "- Practice mindfulness or meditation techniques.",
            "- Engage in hobbies and activities that bring joy and relaxation.",
            "- Limit exposure to stressors and practice stress-reduction techniques.",
            "- Connect with friends, family, or support groups.",
            "- Consider therapy or counseling to develop coping strategies.",
            "",
            "<b>Medications:</b>",
            "- Medication use should always be under medical supervision.",
            "",
            "<b>Professional Help:</b>",
            "- Seek counseling, therapy, or psychiatric help for severe or chronic stress."
        ]

    prompt = f"Provide concise and detailed medical suggestions and lifestyle changes for an individual with a '{stress_level}' stress level. Each section should have 2-3 concise bullet points. Include sections for 'Precautions and Lifestyle Changes', 'Mindfulness and Meditation', 'Sleep Hygiene', 'Physical Exercise', 'Social Support', 'Stress-Related Eating'. If the stress level is 'High Stress' or 'Normal Stress', also include sections for 'Medications' and 'Professional Help'. Format the output with bold headings for each section and bullet points for the suggestions under each heading. Ensure the output is clean and professional for a medical report. Example format: **Heading:**\n- Suggestion 1\n- Suggestion 2\n\n"

    try:
        response = genai_model.generate_content(prompt)
        # The Gemini API response might need parsing depending on its exact structure
        # Assuming the response.text contains the formatted suggestions directly.
        generated_text = response.text
        
        # Split the text into lines and format for the PDF
        formatted_suggestions = []
        for line in generated_text.splitlines():
            stripped_line = line.strip()
            if not stripped_line:
                continue
            
            # First, remove all markdown bolding symbols
            cleaned_line = stripped_line.replace('**', '')
            
            if stripped_line.startswith('**') and stripped_line.endswith('**:'):
                # This is a main heading like **Heading:**
                formatted_suggestions.append(cleaned_line[:-1].strip()) # Remove the trailing colon for clean heading
            elif stripped_line.startswith('-'):
                # This is a bullet point, potentially with bolding inside like - **Subheading:**
                formatted_suggestions.append(cleaned_line)
            else:
                # Any other lines, potentially introductory text or plain paragraphs
                formatted_suggestions.append(cleaned_line)
        return formatted_suggestions
    except Exception as e:
        print(f"Error generating suggestions from GenAI: {e}")
        # Fallback to hardcoded suggestions in case of API error
        return [
            "<b>Precautions and Lifestyle Changes:</b>",
            "- Consult a healthcare professional for a comprehensive evaluation.",
            "- Maintain a balanced diet and regular exercise routine.",
            "- Ensure adequate sleep (7-9 hours per night for adults).",
            "- Practice mindfulness or meditation techniques.",
            "- Engage in hobbies and activities that bring joy and relaxation.",
            "- Limit exposure to stressors and practice stress-reduction techniques.",
            "- Connect with friends, family, or support groups.",
            "- Consider therapy or counseling to develop coping strategies.",
            "",
            "<b>Medications:</b>",
            "- Medication use should always be under medical supervision.",
            "",
            "<b>Professional Help:</b>",
            "- Seek counseling, therapy, or psychiatric help for severe or chronic stress."
        ]


@app.route("/contact")
def contact():
    return render_template('contact.html')

def generate_medical_report_pdf(username, stress_level, stress_percentage, suggestions, filepath):
    doc = SimpleDocTemplate(filepath, pagesize=letter, onFirstPage=header_footer, onLaterPages=header_footer)
    styles = getSampleStyleSheet()

    # Custom styles - ensure unique names or modify existing ones appropriately
    styles.add(ParagraphStyle(name='ReportTitle', fontSize=24, leading=28, alignment=TA_CENTER, spaceAfter=20, fontName='Helvetica-Bold'))
    # Using existing styles directly, or defining custom ones with unique names if needed
    styles.add(ParagraphStyle(name='CustomHeading1', fontSize=18, leading=22, spaceAfter=14, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='CustomHeading2', fontSize=14, leading=18, spaceAfter=10, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='CustomBodyText', fontSize=12, leading=14, spaceAfter=6))
    styles.add(ParagraphStyle(name='CustomFooter', fontSize=9, leading=10, alignment=TA_CENTER, textColor='gray'))
    styles.add(ParagraphStyle(name='CustomRightAlign', fontSize=12, leading=14, alignment=TA_RIGHT))
    styles.add(ParagraphStyle(name='SuggestionHeading', fontSize=12, leading=14, spaceAfter=6, fontName='Helvetica-Bold')) # New style for suggestion headings
    styles.add(ParagraphStyle(name='SuggestionText', fontSize=11, leading=13, spaceAfter=3, leftIndent=20))

    Story = []

    # Title
    Story.append(Paragraph("Medical Report", styles['ReportTitle']))
    Story.append(Spacer(1, 0.2 * inch))

    # Patient Information
    Story.append(Paragraph("Patient Information:", styles['CustomHeading1']))
    Story.append(Paragraph(f"<b>Username:</b> {username}", styles['CustomBodyText']))
    Story.append(Paragraph(f"<b>Date of Report:</b> {datetime.date.today().strftime('%Y-%m-%d')}", styles['CustomBodyText']))
    Story.append(Spacer(1, 0.2 * inch))

    # Stress Analysis Summary
    Story.append(Paragraph("Stress Analysis Summary:", styles['CustomHeading1']))
    Story.append(Paragraph(f"<b>Stress Level:</b> {stress_level}", styles['CustomBodyText']))
    Story.append(Paragraph(f"<b>Stress Percentage:</b> {stress_percentage:.2f}%", styles['CustomBodyText']))
    Story.append(Spacer(1, 0.2 * inch))

    # GenAI Suggestions
    Story.append(Paragraph("Personalized Suggestions:", styles['CustomHeading1']))
    for suggestion_line in suggestions:
        # Check if it's a heading (ends with a colon, not a bullet point)
        if suggestion_line.endswith(':') and not suggestion_line.startswith('-'):
            Story.append(Paragraph(suggestion_line, styles['SuggestionHeading']))
        elif suggestion_line.startswith('-'):
            Story.append(Paragraph(suggestion_line, styles['SuggestionText']))
        else:
            Story.append(Paragraph(suggestion_line, styles['CustomBodyText'])) # For blank lines or other text
    Story.append(Spacer(1, 0.5 * inch))

    # Disclaimer
    Story.append(Paragraph("Disclaimer: This report is generated by an automated system based on emotion detection analysis and general AI suggestions. It is not a substitute for professional medical advice. Always consult with a qualified healthcare provider for diagnosis and treatment.", styles['CustomFooter']))

    doc.build(Story)
    return filepath

def send_email_with_pdf(recipient_email, subject, body, pdf_filepath):
    """
    Sends an email with the generated PDF report as an attachment.
    This function uses placeholder SMTP configuration.
    """

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    # Attach PDF
    with open(pdf_filepath, 'rb') as attachment:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header(
        'Content-Disposition',
        f'attachment; filename= {os.path.basename(pdf_filepath)}'
    )
    msg.attach(part)

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls() # Secure the connection
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print(f"Email sent successfully to {recipient_email}")
        return True
    except Exception as e:
        print(f"Failed to send email to {recipient_email}: {e}")
        return False

@app.route('/')
def home():
    
    return render_template('home.html')

@app.route('/detection') 
@login_required 
def index():
    if session.get('role') == 'admin':
        return redirect(url_for('admin_results'))
    username = session.get('username') # Get username from session
    return render_template('index.html', username=username) # Pass username to template

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = get_user(username)

        if user and check_password_hash(user['password'], password):
            session['username'] = user['username'] # Store username
            session['role'] = user['role']       # Store role
            if user['role'] == 'admin':
                return redirect(url_for('admin_results'))
            else:
                return redirect(url_for('index')) # Changed to 'index'
        else:
            return render_template('login.html', error='Invalid username or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email'] # Get email from form
        password = request.form['password']
        
        if not username or not email or not password:
            return render_template('register.html', error='Username, email and password are required')

        hashed_password = generate_password_hash(password)
        if add_user(username, email, hashed_password):
            session['username'] = username
            session['role'] = 'user' # Default role for new registrations
            return redirect(url_for('index')) 
        else:
            return render_template('register.html', error='Username already exists')
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('role', None) # Clear role from session
    return redirect(url_for('home'))

@app.route('/admin_results')
@admin_required # Protect admin page
def admin_results():
    results = get_all_analysis_results()
    # Sort results by date and time, most recent first
    results.sort(key=lambda x: datetime.datetime.strptime(x['Date & Time'], "%Y-%m-%d_%H-%M-%S"), reverse=True)

    # Preprocess results to extract only the filename for video links
    for result in results:
        if result['Link to saved video proof'] and result['Link to saved video proof'] != 'N/A':
            
            result['video_proof_filename'] = os.path.basename(result['Link to saved video proof'])
        else:
            result['video_proof_filename'] = 'N/A'

        # Summarize emotion sequence for display
        if result['Emotion Sequence'] and result['Emotion Sequence'] != 'N/A':
            emotion_counts = Counter(result['Emotion Sequence'])
            # Format into a readable string: e.g., "Happy (50) Fearful (30)"
            formatted_emotion_sequence = " ".join([f"{emotion.capitalize()} ({count})" for emotion, count in emotion_counts.items()])
            result['Formatted Emotion Sequence'] = formatted_emotion_sequence
        else:
            result['Formatted Emotion Sequence'] = 'N/A'


    return render_template('admin_results.html', results=results)

@app.route('/notify_user/<patient_name>', methods=['POST'])
@admin_required
def notify_user(patient_name):
    print('Sending email..')
    all_results = get_all_analysis_results()
    patient_result = None
    for result in all_results:
        if result.get('Patient Name') == patient_name:
            patient_result = result
            break

    if not patient_result:
        return jsonify({'error': 'Patient analysis results not found.'}), 404

    stress_level = patient_result.get('Stress Level Status', 'Undefined')
    stress_percentage = patient_result.get('Stress Percentage', 0.0)

    # Retrieve user's email address
    user_data = get_user(patient_name)
    if not user_data or 'email' not in user_data:
        return jsonify({'error': f'Email address not found for user: {patient_name}'}), 404
    recipient_email = user_data['email']

    print(f"Attempting to notify user: {patient_name} (Email: {recipient_email}) with Stress Level: {stress_level}, Percentage: {stress_percentage}%")

    # 1. Get GenAI suggestions
    suggestions = get_genai_suggestions(stress_level)

    # 2. Generate PDF report
    pdf_filename = f"{patient_name}_MedicalReport_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    pdf_filepath = os.path.join(REPORTS_FOLDER, pdf_filename)
    
    try:
        generate_medical_report_pdf(patient_name, stress_level, stress_percentage, suggestions, pdf_filepath)
    except Exception as e:
        print(f"Error generating PDF for {patient_name}: {e}")
        return jsonify({'error': 'Failed to generate PDF report.'}), 500

    # 3. Send email with PDF
    email_subject = f"Medical Report for {patient_name} - Stress Level: {stress_level}"
    email_body = (
        f"Dear {patient_name},\n\n"
        "Please find attached your personalized medical report based on your recent emotion detection analysis.\n\n"
        "This report includes your stress level, stress percentage, and tailored suggestions generated by our AI system.\n\n"
        "Disclaimer: This report is generated by an automated system and is not a substitute for professional medical advice. Always consult with a qualified healthcare provider for diagnosis and treatment.\n\n"
        "Sincerely,\n"
        "The Autism & Stress Detection System Team"
    )

    if send_email_with_pdf(recipient_email, email_subject, email_body, pdf_filepath):
        # Clean up the generated PDF file after sending
        os.remove(pdf_filepath)
        return jsonify({'message': f'Medical report sent to {recipient_email} for {patient_name}.'}), 200
    else:
        os.remove(pdf_filepath) # Ensure cleanup even on email failure
        return jsonify({'error': 'Failed to send medical report via email.'}), 500

@app.route('/predict_emotion', methods=['POST'])
@login_required # Protect API endpoints
def predict_emotion():
    data = request.json
    frames_data = data.get('frames')
    patient_name = data.get('patientName', 'Unknown')

    if not frames_data:
        return jsonify({'error': 'No frames provided'}), 400

    decoded_frames = []
    for frame_base64 in frames_data:
        img_bytes = base64.b64decode(frame_base64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        decoded_frames.append(frame)
    
    result = analyze_emotions_from_frames(decoded_frames, patient_name, normal_stress_threshold=NORMAL_STRESS_THRESHOLD, high_stress_threshold=HIGH_STRESS_THRESHOLD)

    return jsonify(result)

@app.route('/predict_emotion_frame', methods=['POST'])
@login_required # Protect API endpoints
def predict_emotion_frame():
    data = request.json
    frame_base64 = data.get('frame')

    if not frame_base64:
        return jsonify({'error': 'No frame provided'}), 400

    try:
        img_bytes = base64.b64decode(frame_base64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Could not decode image frame'}), 400

        result = get_emotion_for_single_frame(frame)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Error processing frame: {e}'}), 500

@app.route('/predict_autism_image', methods=['POST'])
@login_required # Protect API endpoints
def predict_autism_image():
    if AUTISM_MODEL is None or AUTISM_CLASS_NAMES is None:
        return jsonify({'error': 'stress detection model not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        prediction = AUTISM_MODEL.predict(data)
        index = np.argmax(prediction)
        class_name = AUTISM_CLASS_NAMES[index].strip()
        confidence_score = float(prediction[0][index])

        return jsonify({
            'class_name': class_name,
            'confidence_score': confidence_score
        })
    except Exception as e:
        return jsonify({'error': f'Error processing image: {e}'}), 500

@app.route('/predict_autism_video', methods=['POST'])
@login_required # Protect API endpoints
def predict_autism_video():
    patient_name = request.form.get('patientName', 'Unknown')

    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_video_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            result = analyze_emotions_from_video(filepath, patient_name, normal_stress_threshold=NORMAL_STRESS_THRESHOLD, high_stress_threshold=HIGH_STRESS_THRESHOLD)
            # Clean up the uploaded file after processing
            os.remove(filepath)
            return jsonify(result)
        except Exception as e:
            os.remove(filepath) # Ensure cleanup even on error
            return jsonify({'error': f'Error processing video: {e}'}), 500
    else:
        return jsonify({'error': 'Invalid video file type'}), 400

@app.route('/videos/<filename>')
def serve_video(filename):
    return send_from_directory(VIDEO_SAVE_FOLDER, filename)

if __name__ == '__main__':
    from keras.models import load_model 
    app.run(debug=True, host='127.0.0.1', port=5002)
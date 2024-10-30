from flask import Flask, request, render_template, jsonify
from Lesion import InferencePipeline
import tier
import json
import re
import clot
import os
import google.generativeai as genai
import logging
prompt_template = """
Patient Details:
    
    Medical History:
    Currently on any medication or insulin:{medication}
    Changes in Vision:{vision_changes}
    floaters, spots, or flashes of light in your vision:{Floaters}
    Condition: Diabetic Retinopathy classified as {tierss}
    
    Lesion Size: {lesion_area}

    age:{age}

Required Patient Data:
    - Symptoms: {symptoms}

Provide a detailed consultation that includes:
- Explanation of the condition based on {tier} severity.
- Medical treatment options.
- Necessary precautions and lifestyle changes.
- Follow-up frequency and tests required.
- Potential risks and complications.
"""
class_names = ['mild', 'moderate', 'no_dr', 'poliferate', 'severe']  # Get class names from training data

def get_doctor_prompt():
    # Read the doctor prompt from the file
    prompt_file_path = os.path.join(app.static_folder, 'doctorprompt.txt')
    with open(prompt_file_path, 'r') as file:
        doctor_prompt = file.read()
    return doctor_prompt

app = Flask(__name__)

# Set up logging to check backend activity
logging.basicConfig(level=logging.INFO)

# Configure Google Generative AI
genai.configure(api_key="PASTE YOUR GEMINI API KEY HERE")

# Generation configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Define the model and chat session
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)
chat_session = model.start_chat(history=[])
doctor_prompt = get_doctor_prompt()

def get_google_response(prompt):
    try:
     


        # Generate a response using Google Generative AI
        completion = chat_session.send_message(doctor_prompt + prompt)
        # Log the response in the backend to check the integration
        logging.info(f"Google Generative AI Response: {completion.text}")
        return completion.text
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return f"Error: {str(e)}"



modellesionmask = "D:\\testindian\\best_weights_vgg16.keras"
pipeline = InferencePipeline(modellesionmask)
classifier = tier.EnsembleClassifier(weights_path='D:\\testindian\\best_ensemble_model.pth')
detector = clot.ClotDetector()

# @app.route('/chat', methods=['POST'])
# def chat():
#     user_input = request.form['user_input']
#     if user_input:
#         # Call the Google Generative AI function and return the response
#         response = get_google_response(user_input)
#         return jsonify({'response': response})
#     return jsonify({'response': 'No input provided'})

# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'image' not in request.files:
#         return 'No file part'
    
#     file = request.files['image']
    
#     if file.filename == '':
#         return 'No selected file'
    
#     if file:
#         # Ensure uploads folder exists
#         uploads_dir = 'uploads/'
#         os.makedirs(uploads_dir, exist_ok=True)
        
#         # Save the file
#         file_path = os.path.join(uploads_dir, file.filename)
#         file.save(file_path)
#         path="uploads\\"+file.filename
#         fundus_area,masked_area,ratio=pipeline.process_image(path)
        

#         return jsonify({'message': f'File {fundus_area} uploaded successfully at {ratio}', 'filename': file.filename})
    
#     return jsonify({'message': 'File upload failed'})
@app.route('/')
def upload_index():
    return render_template('firsts.html')
@app.route('/submit-report', methods=['POST'])

def submit_report():
    
    # Get form data
    name = request.form.get('name')
    age = request.form.get('age')
    sex = request.form.get('sex')
    blood_pressure = request.form.get('blood_pressure')
    blood_group = request.form.get('blood_group')
    medication = request.form.get('medication')
    vision_changes = request.form.get('vision_changes')
    Floaters=request.form.get('floaters')
    visionpain=request.form.get('eye_pain')
    familyhistory=request.form.get('family_diabetes')
    familyvisionhistory=request.form.get('family_vision_problems')
    additional_comments = request.form.get('message')
    
    # Get the uploaded images
    left_eye_image = request.files['left_eye_image']
    right_eye_image = request.files['right_eye_image']
    

    # Save the uploaded images
    uploads_dir = os.path.join(os.getcwd(), 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)
    
    left_eye_path = os.path.join(uploads_dir, left_eye_image.filename)
    right_eye_path = os.path.join(uploads_dir, right_eye_image.filename)
    left_eye_image.save(left_eye_path)
    right_eye_image.save(right_eye_path)

    
    if not left_eye_image or left_eye_image.filename == '':
        return "Error: No left eye image uploaded."

    if not right_eye_image or right_eye_image.filename == '':
        return "Error: No right eye image uploaded."
    predicted_class, class_name, highest_prob, probabilities = classifier.infer(right_eye_path, class_names)
    
   
    fundus_area, masked_area,ratio=pipeline.process_image(left_eye_path)
    fundus_area, masked_area,ratio2=pipeline.process_image(right_eye_path)
    prediction, confidence = detector.predict(left_eye_path)
    ratio=(ratio+ratio2)/2
    ratio = f"{ratio:.3f}"
    prediction = confidence >= 0.5
    templete=prompt_template.format(
        medication=medication,
        vision_changes=vision_changes,
        Floaters=Floaters,
        lesion_area=ratio,
        age=age,
        tierss=f'Predicted Class: {predicted_class} ({class_name})',
        tier=class_name,
       
        symptoms="asdasd"

    
        
    )
    response=get_google_response(templete)
    test=clean_and_parse_json(response)
    if test:
        logging.info(json.dumps(test, indent=2))
    
    logging.info(f'Predicted Class: {response})')
    
    # Create a response (or process the data)
    response_data = {
        'name':name,
        'age':age,
        'sex':sex,
        'BP':blood_pressure,
        'BD':blood_group,
        'ratio':ratio,
        'nerve':prediction,
        'DMI':True,
        'tier':class_name,
        'analysis':str(test['analysis']),
        'precautions':str(test['precautions']),
        'tests':str(test['tests']),
        'conclusion':str(test['conclusion'])
        
    }
    
    # Send back a JSON response
    return render_template('report.html',data=response_data)

def clean_and_parse_json(malformed_json_string):
    # Remove leading and trailing triple backticks and the "json" after the first set
    clean_string = re.sub(r'^```json\s*|\s*```$', '', malformed_json_string.strip())

    # Attempt to parse the cleaned string
    try:
        json_data = json.loads(clean_string)
        return json_data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
if __name__ == '__main__':
    app.run(debug=True)

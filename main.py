# Import necessary libraries
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os
import face_recognition

# Initialize Flask app
app = Flask(__name__)

# Path to the folder where the face images are stored
faces_folder_path = "faces"

# Function to recognize faces
def recognize_faces(image):
    # Load known faces and their encodings
    known_face_encodings = []
    known_face_names = []
    for filename in os.listdir(faces_folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            name = os.path.splitext(filename)[0]
            image_path = os.path.join(faces_folder_path, filename)
            face_image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(face_image)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(name)

    # Find all the faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    # Loop through each face found in the frame
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match is found, use the name of that person
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        return name

    return None  # Return None if no face is recognized

# Route to display webcam capture for face registration
@app.route('/register_face_cam')
def register_face_cam():
    return render_template('register_face_cam.html')

# Route to handle webcam face registration
@app.route('/register_face_webcam', methods=['POST'])
def register_face_webcam():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow('Register Face', frame)

        # Capture image when 'Space' key is pressed
        if cv2.waitKey(1) & 0xFF == ord(' '):
            # Save the captured image to the faces folder with a unique filename
            filename = f"face_{len(os.listdir(faces_folder_path))}.jpg"
            cv2.imwrite(os.path.join(faces_folder_path, filename), frame)
            break

    cap.release()
    cv2.destroyAllWindows()

    return jsonify({'success': True, 'message': 'Face registered successfully'})

# Route for recognizing attendance
@app.route('/recognize_attendance', methods=['POST'])
def recognize_attendance():
    # Receive image data
    image_data = request.files['image'].read()
    nparr = np.fromstring(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Recognize faces
    name = recognize_faces(image)

    if name is not None:
        return jsonify({'success': True, 'message': f'Attendance marked for: {name}'})
    else:
        return jsonify({'success': False, 'message': 'No face recognized'})

if __name__ == '__main__':
    app.run(debug=True)

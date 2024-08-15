from flask import Flask, render_template, request, jsonify, Response
import cv2
import mediapipe as mp
import numpy as np

# Create a Flask application object
app = Flask(__name__)

counter = 0
stage = None  # Initialize the stage variable

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

desired_width = int(200)
desired_height = int(200)

def generate_face():
    global counter, stage  # Make sure to use the global variables
    
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Recolor
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Detect
            results = pose.process(image)
            
            # Recolor back
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)
                
                # Display angle
                cv2.putText(image, str(angle), 
                            tuple(np.multiply(elbow, [desired_width, desired_height]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 25, 25), 2, cv2.LINE_AA)
                
                # Curl count logic
                if angle > 155:
                    stage = "down"
                if angle < 30 and stage == "down":
                    stage = "up"
                    counter += 1
                
            except:
                pass   
            
            # Display count
            cv2.putText(image, "Count", (15, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 25), 2, cv2.LINE_AA)
            cv2.putText(image, str(counter), (15, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (8,8,8), 2, cv2.LINE_AA)
            
            # Display stage
            cv2.putText(image, "Stage", (200, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 25), 2, cv2.LINE_AA)
            cv2.putText(image, str(stage), (200, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (8,8,8), 2, cv2.LINE_AA)
                        
            # Render landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)  
            
            # Resize the image
            image = cv2.resize(image, (desired_width, desired_height))                  
            
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()
    cv2.destroyAllWindows()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture', methods=['GET', 'POST'])
def capture():
    global capturing_faces
    print(request.method)
    if request.method == 'POST':
        capturing_faces = True
        return render_template('freetrial.html', capturing=True)
    return render_template('freetrial.html', capturing=False)
    
    

@app.route('/video_feed_face')
def video_feed_face():
    return Response(generate_face(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/joinus')
def joinus():
    return render_template("joinus.html")

@app.route("/payment")
def payment():
    return render_template("payment.html")

@app.route("/success")
def success():
    return render_template("successful.html")


if __name__ == '__main__':
    app.run(debug=True)
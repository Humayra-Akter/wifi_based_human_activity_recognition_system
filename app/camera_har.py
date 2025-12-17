import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

def webcam_activity():
    cap = cv2.VideoCapture(0)
    
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            
            # Default activity
            activity = "NO POSE DETECTED"
            
            # Only process if pose landmarks are detected
            if results.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )
                # Classify pose only when landmarks exist
                activity = classify_pose(results.pose_landmarks.landmark)
            
            # Display activity
            cv2.putText(
                frame,
                f"Activity: {activity}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            cv2.imshow("Live HAR (Webcam)", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()


def classify_pose(landmarks):
    """Simple rule-based pose classification"""
    # Hip & knee landmarks
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    
    # Simple vertical comparison
    if left_hip.y < left_knee.y:
        return "STANDING"
    else:
        return "SITTING"
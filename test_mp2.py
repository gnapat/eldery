import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection and Mesh solution
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize webcam feed
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam feed
    ret, frame = cap.read()

    # Convert frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in frame
    results_detection = face_detection.process(image)

    # Extract face landmarks and mesh
    if results_detection.detections:
        face_landmarks = results_detection.detections[0] #.landmarks
        print(face_landmarks)
        results_mesh = face_mesh.process(image, face_landmarks)
        if results_mesh.multi_face_landmarks:
            face_mesh = results_mesh.multi_face_landmarks[0]

    # Visualize face landmarks and mesh
    if face_mesh:
        for landmark in face_landmarks.landmark:
            x, y, z = landmark.x, landmark.y, landmark.z
            # Draw 2D landmark on frame
            cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 2, (0, 255, 0), -1)
        for triangle in face_mesh.triangle_indices:
            # Draw 3D mesh on frame
            pt1 = (int(face_mesh.landmark[triangle[0]].x * frame.shape[1]), int(face_mesh.landmark[triangle[0]].y * frame.shape[0]))
            pt2 = (int(face_mesh.landmark[triangle[1]].x * frame.shape[1]), int(face_mesh.landmark[triangle[1]].y * frame.shape[0]))
            pt3 = (int(face_mesh.landmark[triangle[2]].x * frame.shape[1]), int(face_mesh.landmark[triangle[2]].y * frame.shape[0]))
            cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
            cv2.line(frame, pt2, pt3, (255, 0, 0), 2)
            cv2.line(frame, pt3, pt1, (255, 0, 0), 2)

    # Show frame with face landmarks and mesh
    cv2.imshow('Face Model', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
face_detection.close()
face_mesh.close()
cap.release()
cv2.destroyAllWindows()

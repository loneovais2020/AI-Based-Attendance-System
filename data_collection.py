# import cv2
# import numpy as np 
# import os

# # Load face detection model
# model = r"Attendance system\models\res10_300x300_ssd_iter_140000_fp16.caffemodel"
# config = r"Attendance system\models\deploy.prototxt.txt"
# face_detector_model = cv2.dnn.readNetFromCaffe(config, model)

# # Get the directory of this python script 
# current_dir = os.path.dirname(os.path.abspath(__file__))

# def save_webcam_frames(username):

#     total_frames = 100
#     saved_frames = 0
    
#     # Construct training data path
#     train_data_path = os.path.join(current_dir, 'training data', username)
#     os.makedirs(train_data_path, exist_ok=True)
    
#     cap = cv2.VideoCapture(1)
#     frame_count = 0
    
#     while saved_frames <= total_frames:
        
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         # Only save every 5th frame
#         if frame_count % 5 == 0:
            
#             # Face detection
#             img = frame.copy()
#             h,w = img.shape[:2]
#             img_blob = cv2.dnn.blobFromImage(img, 1 ,(300,300), 
#                                             (104, 177,123), swapRB=False,crop=False)
#             face_detector_model.setInput(img_blob)
#             detections = face_detector_model.forward()
            
#             if len(detections) > 0:
#                 confidence = detections[0,0,0,2]
#                 if confidence > 0.85:
#                     filename = f"frame_{saved_frames}.jpg"
#                     cv2.imwrite(os.path.join(train_data_path, filename), frame)
#                     saved_frames += 1

#         # Display tracking
#         text = f"Saved {saved_frames}/{total_frames}" 
#         cv2.putText(frame, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0),2)
#         cv2.imshow('Webcam', frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
            
#         frame_count += 1
        
#     cap.release()
#     cv2.destroyAllWindows()
    
# if __name__ == '__main__':
#     username = input("Enter your name: ")
#     save_webcam_frames(username)






import cv2
import numpy as np
import os
import streamlit as st

# Load face detection model
model = r"models\res10_300x300_ssd_iter_140000_fp16.caffemodel" 
config = r"models\deploy.prototxt.txt"
face_detector_model = cv2.dnn.readNetFromCaffe(config, model)

# Get the directory of this python script
current_dir = os.path.dirname(os.path.abspath(__file__)) 

st.title("Webcam Face Capture")

username = st.text_input("Enter your name:")

if username:
    st.write("Capturing webcam frames for", username)
    
    total_frames = 100
    saved_frames = 0
    
    # Construct training data path 
    train_data_path = os.path.join(current_dir, 'training data', username)
    os.makedirs(train_data_path, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    frame_count = 0
    
    stframe = st.empty()
    
    while saved_frames <= total_frames:
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Only save every 5th frame
        if frame_count % 5 == 0:
            # Face detection
            img = frame.copy()
            h,w = img.shape[:2]
            img_blob = cv2.dnn.blobFromImage(img, 1 ,(300,300),  
                                             (104, 177,123), swapRB=False,crop=False)
            face_detector_model.setInput(img_blob)
            detections = face_detector_model.forward()
            
            if len(detections) > 0:
                confidence = detections[0,0,0,2]
                if confidence > 0.85:
                    filename = f"frame_{saved_frames}.jpg"
                    cv2.imwrite(os.path.join(train_data_path, filename), frame)
                    saved_frames += 1
            
            # Display tracking 
            text = f"Saved {saved_frames}/{total_frames}"
            cv2.putText(frame, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0),2)
            
            stframe.image(frame, channels="BGR")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
        
    cap.release()
    st.success("ALL IMAGES CAPTURED")
    
    

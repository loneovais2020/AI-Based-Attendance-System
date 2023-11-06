import cv2
import numpy as np
import pickle
import dlib
import math
import numpy as np



# Load models
model = r"face recognition and detection\Attendance system\models\res10_300x300_ssd_iter_140000_fp16.caffemodel"
config = r"face recognition and detection\Attendance system\models\deploy.prototxt.txt"
face_descriptor=r"face recognition and detection\Attendance system\models\openface.nn4.small2.v1.t7"
shape_predictor=r"face recognition and detection\Attendance system\models\shape_predictor_68_face_landmarks.dat"

face_detector_model = cv2.dnn.readNetFromCaffe(config, model)
face_feature_model =cv2.dnn.readNetFromTorch(face_descriptor)
pose_predictor=dlib.shape_predictor(shape_predictor)

face_recognition_model=pickle.load(open(r"D:\LELAFE training\face recognition and detection\Attendance system\My Models\machinelearning_face_person_identity.pickle",mode='rb'))

def detect_side_pose(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h,w = frame.shape[:2]
    # Detect faces
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detector_model.setInput(blob)
    faces = face_detector_model.forward()
    
    # Landmark detection
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence < 0.9:
            continue

        box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x, y, x1, y1) = box.astype("int")
            
        # cv2.rectangle(frame, (x,y),(x1,y1),(0,0,255), 2)
            
        # Detect landmarks
        face = dlib.rectangle(left=x, top=y, right=x1, bottom=y1)
        landmarks = pose_predictor(frame_rgb, face)


        # Get eye landmarks
        left_eye = landmarks.part(36) 
        right_eye = landmarks.part(45)

        # Get center points of eyes
        left_eye_center = (left_eye.x, left_eye.y)
        right_eye_center = (right_eye.x, right_eye.y)

        # Calculate slope and angle between eyes
        dy = right_eye_center[1] - left_eye_center[1] 
        dx = right_eye_center[0] - left_eye_center[0]
        angle = math.atan2(dy, dx) # in radians


        # Determine pose based on angle
        POSE_THRESH = 0.2 # radian threshold
        if abs(angle) > POSE_THRESH:
            
            return True
        else:
            return False




def is_far(face_width):
     

     #do check the unit of the face_Width . is it pixels or is it mm or anything else.



     area_threshold=3900
     f_area=face_width*face_width
     print(f"the area of face is {f_area}")
     if f_area<area_threshold:
         return True
     else:
         return False
     


  
       
def handle_unknown_faces(face_encoding, threshold,pose,face_width):

  # Get prediction from model
  face_name = face_recognition_model.predict(face_encoding)[0]
  face_score = face_recognition_model.predict_proba(face_encoding).max()
  
 
  
  # Check if probability is less than threshold
  if face_score < threshold or pose or is_far(face_width):
    # If so, return 'unknown'
    face_score=1
    face_name="UNKNOWN"
    return face_name,face_score
  
  # Otherwise, return predicted name
  return face_name,face_score



def run_recognition():

    # open external webcam
    cap = cv2.VideoCapture(1)

    # open webcam
    # cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # pipeline model
        img = frame.copy()
        h,w = img.shape[:2]
        # face detection
        img_blob = cv2.dnn.blobFromImage(img,1,(300,300),(104,177,123),swapRB=False,crop=False)
        face_detector_model.setInput(img_blob)
        detections = face_detector_model.forward()
        
        
        count = 1
        if len(detections) > 0:
            for i , confidence in enumerate(detections[0,0,:,2]):
                if confidence > 0.95:
                    box = detections[0,0,i,3:7]*np.array([w,h,w,h])
                    startx,starty,endx,endy = box.astype(int)

                    cv2.rectangle(frame,(startx,starty),(endx,endy),(0,255,0))

                    # feature extraction
                    face_roi = img[starty:endy,startx:endx]
                    face_blob = cv2.dnn.blobFromImage(face_roi,1/255,(96,96),(0,0,0),swapRB=True,crop=True)
                    face_feature_model.setInput(face_blob)
                    vectors = face_feature_model.forward()
                    face_width = endx - startx
                    

                    #predict poses
                    pose=detect_side_pose(frame)
                    # predict with machine learning
                    face_name,face_score = handle_unknown_faces(face_encoding=vectors, threshold=0.80,pose=pose,face_width=face_width)
                    # print(face_name)
                    # print(face_score)


                    if face_name!="UNKNOWN":
                        print(face_name)
                        print(face_score)
                    
                    
                    text_face = '{} : {:.0f} %'.format(face_name,100*face_score)
                    cv2.putText(frame,text_face,(startx,starty),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),1)

                
                    
                
                
        
        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


run_recognition()
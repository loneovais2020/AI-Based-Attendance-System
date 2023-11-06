import cv2
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle

# Face detection model
model = r"Attendance system\models\res10_300x300_ssd_iter_140000_fp16.caffemodel"
config = r"Attendance system\models\deploy.prototxt.txt"
detector = cv2.dnn.readNetFromCaffe(config, model)

# Face feature extractor model
face_descriptor = r"Attendance system\models\openface.nn4.small2.v1.t7"
descriptor = cv2.dnn.readNetFromTorch(face_descriptor)

def process_images(image_dir):

  embeddings = []
  names = [] 
  
  folders = os.listdir(image_dir)
  
  for folder in folders:
    folder_path = os.path.join(image_dir, folder)
    images = os.listdir(folder_path)
    
    for image in images:
      image_path = os.path.join(folder_path, image)
      
      vector = extract_face(image_path, detector, descriptor)
      
      if vector is not None:
        embeddings.append(vector)
        names.append(folder)
        
  output = process_face_vectors(names, embeddings)
  
  return output

def extract_face(image_path, detector, descriptor):
  
  img = cv2.imread(image_path)
  image = img.copy()
  h, w = image.shape[:2]
  
  image_blob = cv2.dnn.blobFromImage(image, 1.0, (300,300), [104,117,123], False, False)

  detector.setInput(image_blob)
  detections = detector.forward()

  if len(detections) > 0:

    i = np.argmax(detections[0,0,:,2])
    confidence = detections[0,0,i,2]

    if confidence > 0.80:

      box = detections[0, 0, i, 3:7] * np.array([w,h,w,h])
      startx, starty, endx, endy = box.astype('int')

      roi = image[starty:endy, startx:endx]
      face_blob = cv2.dnn.blobFromImage(roi, 1/255, (96,96), (0,0,0), swapRB=True, crop=True)

      descriptor.setInput(face_blob)
      vector = descriptor.forward()
      print("Feature Extracted Successfully")

      return vector

  return None
      



def process_face_vectors(names, embeddings, threshold=0.7):

  name_to_vectors = {}
  
  for name, embed in zip(names, embeddings):
    if name not in name_to_vectors:
      name_to_vectors[name] = []
    name_to_vectors[name].append(embed)

  output = {'data':[], 'label':[]}
  
  for name, vectors in name_to_vectors.items():

    # Flatten vectors
    vectors_2d = [vec.reshape(128) for vec in vectors]

    filtered = []
    unknown = []
    
    for i, v1 in enumerate(vectors_2d):
      scores = []
      for j, v2 in enumerate(vectors_2d):
        if i != j:
          score = cosine_similarity(v1.reshape(1,-1), v2.reshape(1,-1))[0][0]
          scores.append(score)

      avg_score = np.mean(scores)  
      
      if avg_score >= threshold:
        filtered.append(vectors[i])
      else:
        unknown.append(vectors[i])

    for vec in filtered:
      output["data"].append(vec)
      output["label"].append(name)

    for vec in unknown:
      output["data"].append(vec)
      output["label"].append("unknown")


  return output



# Example Usage
image_dir = 'Attendance system\processed images' 
output = process_images(image_dir)


#checking the classes
print("Employee Data Collection Information")
print(pd.Series(output['label']).value_counts())



# save the data into a pickle file for later use




# Get the directory of this python file
current_dir = os.path.dirname(os.path.abspath(__file__)) 

# Folder name to create  
new_folder = 'My Models'

# Path to new folder
new_folder_path = os.path.join(current_dir, new_folder)

if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)

# pickle.dump(output, open(os.path.join(new_folder_path, "data_face_features.pickle"), "wb"))
pickle.dump(output, open(os.path.join(new_folder_path, "data_face_features.pickle"), "wb"))


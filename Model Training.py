import cv2
import os
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from sklearn.model_selection import train_test_split


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report,accuracy_score,f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV





data=pickle.load(open(r"Attendance system\My Models\data_face_features.pickle",mode="rb"))

X=np.array(data["data"]) #Independent Var
y=np.array(data["label"]) #Dependent Var

X=X.reshape(-1,128) 


x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=0)







def get_report(model, x_train, x_test, y_train, y_test):

  y_pred_train = model.predict(x_train) 
  y_pred_test = model.predict(x_test)

  # Accuracy scores
  acc_train = accuracy_score(y_train, y_pred_train)
  acc_test = accuracy_score(y_test, y_pred_test)

  # F1 scores
  f1_train = f1_score(y_train, y_pred_train, average="macro")
  f1_test = f1_score(y_test, y_pred_test, average="macro")

  print(f"Training Accuracy: {acc_train}")
  print(f"Testing Accuracy: {acc_test}")

  print(f"Training F1 Score: {f1_train}")
  print(f"Testing F1 Score: {f1_test}")

  # Confusion matrices
  cm_train = confusion_matrix(y_train, y_pred_train)
  cm_test = confusion_matrix(y_test, y_pred_test)

  print("Confusion Matrix (Training):")
  print(cm_train)

  print("Confusion Matrix (Testing):") 
  print(cm_test)
  
  # Classification report
  print("Classification Report:")
  print(classification_report(y_test, y_pred_test))





model_voting=VotingClassifier(estimators=[
    ("logistic",LogisticRegression()),
    ("svm",SVC(probability=True)),
    ("rf",RandomForestClassifier()),
    ("knn",KNeighborsClassifier())
],
voting='soft',
weights=[3,4,1,2])


model_voting.fit(x_train,y_train)


get_report(model_voting,x_train,x_test,y_train,y_test)





model_grid=GridSearchCV(model_voting,
                        param_grid={
                            "svm__C":[3,5,7,10],
                            "svm__gamma":[0.1,0.3,0.5],
                            "rf__n_estimators":[5,10,20],
                            "rf__max_depth":[3,5,7],
                            # "knn__n_neighbors":[5,7,9,11,13,15],
                            # "knn__weights":['uniform','distance'],
                            "voting":["soft","hard"]
                        }
                        ,scoring="accuracy",cv=3,n_jobs=1,verbose=2
                        )


model_grid.fit(x_train,y_train)




get_report(model_grid,x_train,x_test,y_train,y_test)


model_best_estimator=model_grid.best_estimator_


#save the model as a pickle file



# Get the directory of this python file
current_dir = os.path.dirname(os.path.abspath(__file__)) 

# Folder name to create  
new_folder = 'My Models'

# Path to new folder
new_folder_path = os.path.join(current_dir, new_folder)

if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)

pickle.dump(model_best_estimator, open(os.path.join(new_folder_path, "machinelearning_face_person_identity.pickle"), "wb"))
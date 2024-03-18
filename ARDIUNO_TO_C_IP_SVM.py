import joblib
import pandas as pd
import  numpy as np
from skimage.feature.texture import graycomatrix
from skimage.feature.texture import graycoprops
from tqdm import tqdm 
import os
from sklearn.preprocessing import MinMaxScaler
import cv2
from sklearn.svm import SVC
import serial
import time


def get_feature(img):
    
    img_graymatrix = graycomatrix(img, [1], [0, np.pi/2])
    # print("img_graymatrix shape: ",img_graymatrix.shape)
    
    img_contrast = graycoprops(img_graymatrix, 'contrast')
    # print("img_contrast shape: ", img_contrast.shape)
    
    img_homogeneity = graycoprops(img_graymatrix, 'homogeneity')
    # print("img_homogeneity shape: ", img_homogeneity.shape)
    
    img_correlation = graycoprops(img_graymatrix, 'correlation')
    # print("img_correlation shape: ", img_correlation.shape)
    
    
    img_contrast_flattened = img_contrast.flatten()
    img_homogeneity_flattened = img_homogeneity.flatten()
    img_correlation_flattened = img_correlation.flatten()
    
    
    features = np.concatenate([img_contrast_flattened, img_homogeneity_flattened, 
                             img_correlation_flattened])
    # print("final_feature shape: ",features.shape)
    
    return(features)

def feature_extraction(path_to_folder, class_label):
    data_list=[]
#     count=1
    for file_name in tqdm(os.listdir(path_to_folder)):
#         if(count>1):
#             break
        path_to_img = os.path.join(path_to_folder,file_name)
        img = cv2.imread(path_to_img, 0) # grayscale image
         
        if np.shape(img) == ():
            continue
        
        final_feature = get_feature(img)
        # print("final_feature shape: ",final_feature.shape)
        # print("final_feature is: ",final_feature)
        
        final_feature=list(final_feature)
        final_feature.insert(0,file_name)
        final_feature.insert(1,class_label)
        
        data_list.append(final_feature)
        
#         count+=1
       
    return(data_list)

# Initialize serial port (change 'COM6' to your port and 9600 to match Arduino baud rate)
ser = serial.Serial('COM6', 9600)
time.sleep(2)  # Wait for serial to initialize


# Load the saved SVM model from the file
model_SVC = joblib.load('svm_model.joblib')

# Now 'loaded_svm_model' contains the trained SVM model, and you can use it for predictions.
nonfog_path1 = "NON_FOGTEST_SVM"

data_list3 = feature_extraction(nonfog_path1, 0)
df3 = pd.DataFrame(data_list3)
df3.shape
array1=df3.values
img_feature=array1[:,2:]

#Extracting the labels from 1st coloumn only as integer 
y_label1=array1[:,1].astype('int')
# print(img_feature.shape)
# print(y_label1.shape)
# Normalise the data after splitting to avoid information leak between train and test set.

scaler_norm1 = MinMaxScaler()
img_feature = scaler_norm1.fit_transform(img_feature)
pred=model_SVC.predict(img_feature)
# print(pred)

img = cv2.imread(r"A:\MINI_PROJECT_1\NON_FOGTEST_SVM\nonfog2.jpeg")
img = cv2.resize(img,(500,500))
# Display the image in a window named "Image"
cv2.imshow("Image", img)

if (pred < 0.5 ):
    print("Clean")
    ser.write(b'0')  # Send '0' to Arduino
    time.sleep(1)    # Wait for 1 second
    response = ser.readline().decode('utf-8').strip()  # Read the response from Arduino
    print(f"Arduino Response: {response}")  # Print the response from Arduino
    
else:
    print("Foggy")
    ser.write(b'1')  # Send '1' to Arduino
    time.sleep(1)    # Wait for 1 second
    response = ser.readline().decode('utf-8').strip()  # Read the response from Arduino
    print(f"Arduino Response: {response}")  # Print the response from Arduino
  
# Break the loop if 'q' is pressed
if cv2.waitKey(0) & 0xFF == ord('q'):
    ser.write(b'0')  # Send '0' to Arduino
    #time.sleep(1)    # Wait for 1 second
    response = ser.readline().decode('utf-8').strip()  # Read the response from Arduino
    print(f"Arduino Response: {response} User Told To Stop The Execution")
           
ser.close()
# Wait for a key press and then close the window
cv2.destroyAllWindows()
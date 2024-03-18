# 1 -- Clean
# 0 -- Fog
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import serial
import time

# Initialize serial port (change 'COM6' to your port and 9600 to match Arduino baud rate)
ser = serial.Serial('COM6', 9600)
time.sleep(2)  # Wait for serial to initialize

# Load the model
model = tf.keras.models.load_model('Fog_NonFoggy2.h5')

# Function to preprocess an image
def preprocess_image(img):
    img = cv2.resize(img, (150, 150))
    img = img.reshape(1, 150, 150, 3)
    return img

# Path to the video file

video_path = 'foggy-v-ip.mp4'

# Open the webcam
cap = cv2.VideoCapture(video_path)



while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    frame = cv2.resize(frame,(500,500))
    # Display the resulting frame
    cv2.imshow('Original Frame', frame)

    # Preprocess the frame for prediction
    
    processed_frame = preprocess_image(frame)

    # Make prediction
    prediction = model.predict(processed_frame)

    # Print or display the prediction (modify this part based on your model's output)
    if prediction > 0.5:
       # print("Clean")
        output = 0
        ser.write(b'0')  # Send '0' to Arduino
        #time.sleep(1)    # Wait for 1 second
        response = ser.readline().decode('utf-8').strip()  # Read the response from Arduino
        print(f"Arduino Response: {response}")  # Print the response from Arduino
        
        
    else:
        #print("Foggy")
        output = 1
        ser.write(b'1')  # Send '1' to Arduino
        #time.sleep(1)    # Wait for 1 second
        response = ser.readline().decode('utf-8').strip()  # Read the response from Arduino
        print(f"Arduino Response: {response}")  # Print the response from Arduino
        
        
        
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        ser.write(b'0')  # Send '0' to Arduino
        #time.sleep(1)    # Wait for 1 second
        response = ser.readline().decode('utf-8').strip()  # Read the response from Arduino
        print(f"Arduino Response: {response} User Told To Stop The Execution")
        break

# Release the capture and destroy all windows
ser.close()
cap.release()
cv2.destroyAllWindows()
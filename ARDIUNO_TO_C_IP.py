# 1 -- Clean
# 0 -- Fog
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import serial
import time

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize serial port (change 'COM6' to your port and 9600 to match Arduino baud rate)
ser = serial.Serial('COM6', 9600)
time.sleep(2)  # Wait for serial to initialize

model = tf.keras.models.load_model('MYMODEL')

# Function to preprocess an image
def preprocess_image(img):
    img = cv2.resize(img, (150, 150))
    img = img.reshape(1, 150, 150, 3)
    return img


# test_image = cv2.imread('FOG_Test_Input.jpg')
test_image = cv2.imread(r'A:\ML\MINI-PROJECT\clean-ip.jpeg')
plt.imshow(test_image)
test_image.shape
test_image = cv2.resize(test_image,(150,150))
test_image.shape
test_image = test_image.reshape(1,150,150,3)


prob = model.predict(test_image)
print(prob)
if (prob > 0.5 ):
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
    
ser.close()
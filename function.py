#import dependency
import cv2
import numpy as np
import os
import mediapipe as mp
# Initialize MediaPipe modules for drawing and hand detection
mp_drawing = mp.solutions.drawing_utils # Drawing utilities for landmarks
mp_drawing_styles = mp.solutions.drawing_styles # Styles for hand landmarks
mp_hands = mp.solutions.hands# Hands module for detecting hand landmarks

# Function to perform hand detection with MediaPipe
def mediapipe_detection(image, model):
    # Convert the image from BGR to RGB for MediaPipe processing
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results
    
# Function to draw landmarks on detected hands with specified styles
def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:# Check if hand landmarks are detected
      for hand_landmarks in results.multi_hand_landmarks:# Loop through each detected hand
            # Draw hand landmarks and connections with predefined styles
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS, # Draw hand connections
            mp_drawing_styles.get_default_hand_landmarks_style(), # Landmark style
            mp_drawing_styles.get_default_hand_connections_style()) # Connection style

# Function to extract keypoints from detected hand landmarks
def extract_keypoints(results):
    if results.multi_hand_landmarks:# Check if hand landmarks are detected
      for hand_landmarks in results.multi_hand_landmarks:# Loop through each detected hand
            # Extract and flatten (x, y, z) coordinates of 21 landmarks
        rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten() if hand_landmarks else np.zeros(21*3) # Return zeros if no landmarks
        return(np.concatenate([rh])) # Return the flattened keypoints as a single array
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

actions = np.array(['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])
# Set the number of sequences (videos or samples per action)
no_sequences = 30
# Define the length of each sequence (number of frames per sequence)
sequence_length = 30

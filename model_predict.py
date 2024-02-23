"""
Group Project(CSC746)
Project Title: Exam Proctoring System Using Face Detection
by Rajini Chittimalla, Sujung Choi, Madhu Sai Vineel Reka
File Description: This code is used to predict the facial landmarks using the trained model.
It includes the code to count the number of times the person is looking at the right or left, duration of each look, and the timestamps of each look.
"""
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
from datetime import datetime
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model

def generate_model(input_sz, num_classes):
    """
    # generate_model() function is used to generate the model to predict the facial landmarks
    #
    # Input
    ----------------
    # input_sz: input size
    # num_classes: number of classes
    #
    # Output
    ----------------
    # nn_model: model to predict the facial landmarks
    """
    inputs = Input(shape=input_sz)    
    L1  = Dense(400, activation = 'relu')(inputs)
    L2  = Dense(200, activation = 'relu')(L1)
    L3  = Dense(100, activation = 'relu')(L2)
    L4  = Dense(50, activation = 'relu')(L3)
    L5  = Dense(32, activation = 'relu')(L4)
    L6  = Dense(16, activation = 'relu')(L5)
    L7  = Dense(num_classes, activation='softmax')(L6)
        
    nn_model = Model(inputs=inputs, outputs=L7)
    nn_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    nn_model.summary()
        
    return nn_model

def get_input(re, le, nt, mc, ret, let):
  """
  get_input() function is used to return the input for the model in the form of numpy array
  #
  # Input
  ----------------
  # re: right eye
  # le: left eye
  # nt: nose tip
  # mc: mouth center
  # ret: right ear tragion
  # let: left ear tragion
  #
  # Output
  ----------------
  # numpy array of the input
  """
  return np.array([[re.x, re.y, le.x, le.y, nt.x, nt.y, mc.x, mc.y, ret.x, ret.y, let.x, let.y]])

def initialize_variables():
    """
    # Initialize_variables() function is to initialize variables.
    #
    # Output
    ----------------
    # dir_record: empty list to store the direction record
    # pre_time: previous time set to None
    # previous_output: previous output set to 0
    # output: current output set to 0
    """
    return [], None, 0, 0

def detect_face(cap, face_detection):
    """
    # detect_face() function is to detect face landmarks using MediaPipe Face Detection.
    #
    # Input
    ----------------
    # cap: webcam input
    # face_detection: face detection model
    #
    # Output
    ----------------
    # results: face detection results
    # image: image with face detection annotations
    """
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        return None

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return results, image

def get_facial_landmarks(record, mp_face_detection):
    """
    get_facial_landmarks() function is to extract facial landmarks from a face detection record.
    #
    # Input
    ----------------
    # record: face detection record
    # mp_face_detection: MediaPipe Face Detection model
    #
    # Output
    ----------------
    # RIGHT_EYE: right eye
    # LEFT_EYE: left eye
    # NOSE_TIP: nose tip
    # MOUTH_CENTER: mouth center
    # RIGHT_EAR_TRAGION: right ear tragion
    # LEFT_EAR_TRAGION: left ear tragion
    """
    RIGHT_EYE = mp_face_detection.get_key_point(record, mp_face_detection.FaceKeyPoint.RIGHT_EYE)
    LEFT_EYE = mp_face_detection.get_key_point(record, mp_face_detection.FaceKeyPoint.LEFT_EYE)
    NOSE_TIP = mp_face_detection.get_key_point(record, mp_face_detection.FaceKeyPoint.NOSE_TIP)
    MOUTH_CENTER = mp_face_detection.get_key_point(record, mp_face_detection.FaceKeyPoint.MOUTH_CENTER)
    RIGHT_EAR_TRAGION = mp_face_detection.get_key_point(record, mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION)
    LEFT_EAR_TRAGION = mp_face_detection.get_key_point(record, mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION)
    
    return RIGHT_EYE, LEFT_EYE, NOSE_TIP, MOUTH_CENTER, RIGHT_EAR_TRAGION, LEFT_EAR_TRAGION

def record_direction(dir_record, pre_time, previous_output, output, RIGHT_COUNT, LEFT_COUNT):
    """
    record_direction() function is to record the time and count for specific head movements.
    #
    # Input
    ----------------
    # dir_record: list to store the direction record
    # pre_time: previous time
    # previous_output: previous output
    # output: current output
    # RIGHT_COUNT: count of looking at the right
    # LEFT_COUNT: count of looking at the left
    #
    # Output
    ----------------
    # dir_record: updated list to store the direction record
    # pre_time: updated previous time
    # RIGHT_COUNT: updated count of looking at the right
    # LEFT_COUNT: updated count of looking at the left
    """
    if previous_output != output:
        if previous_output == 0:
            pre_time = datetime.now()
          
        elif pre_time and previous_output == 1:
            new_time = datetime.now()
            right_duration = (new_time - pre_time).total_seconds()
            
            if right_duration > 1:
                dir_record.append([pre_time.strftime("%H:%M:%S"), new_time.strftime("%H:%M:%S"), \
                                   right_duration, "Right"])
                RIGHT_COUNT += 1
                print(f"Right Count = {RIGHT_COUNT} and Left Count = {LEFT_COUNT}")
            
        elif pre_time and previous_output == 2:
            new_time = datetime.now()
            left_duration = (new_time - pre_time).total_seconds()
            
            if left_duration > 1:
                dir_record.append([pre_time.strftime("%H:%M:%S"), new_time.strftime("%H:%M:%S"), \
                                   left_duration, "Left"])
                LEFT_COUNT += 1
                print(f"Right Count = {RIGHT_COUNT} and Left Count = {LEFT_COUNT}")
    return dir_record, pre_time, RIGHT_COUNT, LEFT_COUNT

def main():
  """
  main() function is to call the functions to 1) generate the model, 2) load the pre-trained weights,
  3) detect the face, 4) extract the facial landmarks, 5) record the direction and time, 
  and 6) save the record results to a csv file.
  """

  # importing the necessary modules from mediapipe
  mp_face_detection = mp.solutions.face_detection
  mp_drawing = mp.solutions.drawing_utils

  # Drawing specification for different directions (straight, right, left)
  straight_spec = mp.solutions.drawing_utils.DrawingSpec(color=(224, 224, 224))
  right_spec = mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255))
  left_spec = mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0))

  # set the number of classes and input size
  num_classes = 3
  input_size = 12

  # initialize counts for right and left
  LEFT_COUNT = 0
  RIGHT_COUNT = 0 

  # generate the model and load the pre-trained weights into the model
  model = generate_model(input_size, num_classes)
  model.load_weights("models/class_monitor_3class_1.h5")

  # Initialize variables
  dir_record, pre_time, previous_output, output = initialize_variables()
    
  # For webcam input:
  cap = cv2.VideoCapture(0)
    
    # Load face detection model
  with mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        RIGHT_COUNT, LEFT_COUNT = 0, 0
        
        while cap.isOpened():
            results, image = detect_face(cap, face_detection)
            
            if results and results.detections:
                for detection in results.detections:
                    record = detection
                    facial_landmarks = get_facial_landmarks(record, mp_face_detection)
                    
                    if all(facial_landmark is not None for facial_landmark in facial_landmarks):
                        input_data = get_input(*facial_landmarks)
                        output = model.predict(input_data, verbose=False)
                        output = np.argmax(output)
                        
                        dir_record, pre_time, RIGHT_COUNT, LEFT_COUNT = record_direction(
                            dir_record, pre_time, previous_output, output, RIGHT_COUNT, LEFT_COUNT
                        )
                        
                        if output == 0:
                            mp_drawing.draw_detection(image, detection, straight_spec)
                        elif output == 1:
                            mp_drawing.draw_detection(image, detection, right_spec)
                        elif output == 2:
                            mp_drawing.draw_detection(image, detection, left_spec)
            
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
            
            previous_output = output  # Update previous_output for the next iteration
            
  cap.release()
  df = pd.DataFrame(dir_record)

  # Save the record results to a csv file
  df.to_csv("record.csv")

main()
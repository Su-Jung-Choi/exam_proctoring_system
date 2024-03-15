"""
Group Project (CSC746)
Project Title: Exam Proctoring System Using Face Detection
by Rajini Chittimalla, Sujung Choi, Madhu Sai Vineel Reka
File Description: This file is to collect the facial landmarks dataset using mediapipe library.
"""
import pandas as pd
import cv2
import mediapipe as mp


def create_csv(collected_data_list, file_name):
  """
  # create_csv() function is used to create csv files to save the facial landmarks using pandas dataframe
  #
  # Input
  # ----------------
  # collected_data_list: list of facial landmarks
  # file_name: file name to save the facial landmarks
  """
  cs = ["right_eye_x", "right_eye_y", "left_eye_x", "left_eye_y", "nose_tip_x", \
        "nose_tip_y", "mouth_center_x", "mouth_center_y", "right_ear_tragion_x",\
          "right_ear_tragion_y", "left_ear_tragion_x", "left_ear_tragion_y"]
  df = pd.DataFrame(collected_data_list, columns=cs)
  df.to_csv(file_name)


def add(re, le, nt, mc, ret, let, collected_data_list):
  """
  # add() function is used to append the facial landmarks to the list
  #
  # Input
  # ----------------
  # re: right eye
  # le: left eye
  # nt: nose tip
  # mc: mouth center
  # ret: right ear tragion
  # let: left ear tragion
  # collected_data_list: list of facial landmarks
  """
  collected_data_list.append([re.x, re.y, le.x, le.y, nt.x, nt.y, mc.x, mc.y, ret.x, ret.y, let.x, let.y])

def main():
  
  mp_face_detection = mp.solutions.face_detection
  mp_drawing = mp.solutions.drawing_utils

  # list of files to save the facial landmarks for each action
  files = ["normal.csv", "right.csv", "left.csv"]

  #replace idx with numbers from 0 to 2 to record the data for each file
  idx = 2
  file_name = files[idx]
  data_size = 2000

  START = False
  print(f"Recording {file_name} data!!!")

  collected_data_list = []

  # For webcam input:
  cap = cv2.VideoCapture(0)
  record = []
  with mp_face_detection.FaceDetection(
      model_selection=0, min_detection_confidence=0.5) as face_detection:
      i = 0
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
          for detection in results.detections:
            record = detection
            mp_drawing.draw_detection(image, detection)
            
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
        
        # check if the data is recorded for the required size 
        if len(collected_data_list) == data_size:
          # create the csv file with the collected data and specified file name
          create_csv(collected_data_list, file_name)
          break # break once the collection is done

        # check for the spacebar (ASCII code 32) to start recording
        if cv2.waitKey(5) & 0xFF == 32:
          print("Recordings is Started!!!")
          START=True
        # check for the escape key (ASCII code 27) to stop the recording
        if cv2.waitKey(5) & 0xFF == 27:
          break

        # extract facial key points using mediapipe face detection module
        RIGHT_EYE = mp_face_detection.get_key_point(record, mp_face_detection.FaceKeyPoint.RIGHT_EYE)
        LEFT_EYE = mp_face_detection.get_key_point(record, mp_face_detection.FaceKeyPoint.LEFT_EYE)
        NOSE_TIP = mp_face_detection.get_key_point(record, mp_face_detection.FaceKeyPoint.NOSE_TIP)
        MOUTH_CENTER = mp_face_detection.get_key_point(record, mp_face_detection.FaceKeyPoint.MOUTH_CENTER)
        RIGHT_EAR_TRAGION = mp_face_detection.get_key_point(record, mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION)
        LEFT_EAR_TRAGION = mp_face_detection.get_key_point(record, mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION)
        
        # check if all key points are detected and the recording is started
        # then print the length of collected data list and add the key points to the list
        if RIGHT_EYE and LEFT_EYE and \
           NOSE_TIP and MOUTH_CENTER and \
           RIGHT_EAR_TRAGION and LEFT_EAR_TRAGION and START:
              print(len(collected_data_list))
              add(RIGHT_EYE, LEFT_EYE, NOSE_TIP, MOUTH_CENTER, RIGHT_EAR_TRAGION, LEFT_EAR_TRAGION, \
                  collected_data_list)

  cap.release()

main()
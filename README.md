# HCI Project - Exam Proctoring System Using Face Detection
This project was completed in Spring 2023 as a final group project for the HCI course. The objective was to build an Exam Proctoring System Using Face Detection. We utilized Mediapipe, OpenCV, Tensorflow, and Keras.

***Following is the list of files contained in this folder:**
1. face_detection_data_collection.py: this file is to collect the data using a Webcam for each right, left, and normal look.
2. split_dataset.py: this file is to split the dataset into train set (80%) and test set (20%).
3. exam_monitoring_train.py: This file is for training the Neural Network model and testing the model with unseen data.
4. model_predict.py: this file is to evaluate the performance of the trained model in predicting and classifying the student's behavior using a live Webcam.


![summary_model](https://github.com/Su-Jung-Choi/exam_proctoring_system/assets/88897881/9d9e45f1-eef8-423f-b39b-39f953ba5340)

Fig 1. Summary of the model

![new_training_result](https://github.com/Su-Jung-Choi/exam_proctoring_system/assets/88897881/376c759f-9959-4c74-b5c2-2b93026ce027)


Fig 2. Accuracy result by testing the trained model

![image](https://github.com/Su-Jung-Choi/exam_proctoring_system/assets/88897881/1bc5c28a-db31-422e-b600-0cee30444164)

Fig 3. Collected record based on model prediction

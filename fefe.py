import face_recognition
import cv2
import numpy as np

# This is a super simple (but slow) example of running face recognition on live video from your webcam.
# There's a second example that's a little more complicated but runs faster.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.
video_capture = cv2.VideoCapture("videos/test_out_04.avi")
length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
# Get a reference to webcam #0 (the default one)
one_image = cv2.imread("dataset/Boldysheva Ekaterina/IMG_20190513_135904.jpg")
one_face_encoding = face_recognition.face_encodings(one_image)[0]

two_image = cv2.imread("dataset/Ostapenko Kirill/IMG_20190513_135742.jpg")
two_face_encoding = face_recognition.face_encodings(two_image)[0]

three_image = cv2.imread("dataset/Balako Alexey/IMG_20190513_135807.jpg")
three_face_encoding = face_recognition.face_encodings(three_image)[0]

four_image = cv2.imread("dataset/Danilov Maxim/IMG_20190513_140057.jpg")
four_face_encoding = face_recognition.face_encodings(four_image)[0]

five_image = cv2.imread("dataset/Griboedov Nikita/IMG_20190513_135642.jpg")
five_face_encoding = face_recognition.face_encodings(five_image)[0]

six_image = cv2.imread("dataset/Simakov Stepan/IMG_20190513_140017.jpg")
six_face_encoding = face_recognition.face_encodings(six_image)[0]

seven_image = cv2.imread("dataset/Machalov Andrey/IMG_20190513_135834.jpg")
seven_face_encoding = face_recognition.face_encodings(seven_image)[0]

eight_image = cv2.imread("dataset/Parakhin Nikita/IMG_20190513_140033.jpg")
eight_face_encoding = face_recognition.face_encodings(eight_image)[0]

nine_image = cv2.imread("dataset/Sherbakov Danil/IMG_20190513_135730.jpg")
nine_face_encoding = face_recognition.face_encodings(nine_image)[0]

ten_image = cv2.imread("dataset/Sokolnikov Egor/IMG_20190513_135936.jpg")
ten_face_encoding = face_recognition.face_encodings(ten_image)[0]

lmm_image = cv2.imread("dataset/Makarov Alexander/IMG_20190513_135954.jpg")
lmm_face_encoding = face_recognition.face_encodings(lmm_image)[0]

known_face_encodings = [
    one_face_encoding,
    two_face_encoding,
    three_face_encoding,
    four_face_encoding,
    five_face_encoding,
    six_face_encoding,
    seven_face_encoding,
    eight_face_encoding,
    nine_face_encoding,
    ten_face_encoding,
    lmm_face_encoding
]

known_face_names = [
    'Boldysheva Ekaterina',
    'Ostapenko Kirill',
    'Balako Alexey',
    'Danilov Maxim',
    'Griboedov Nikita',
    'Simakov Stepan',
    'Machalov Andrey',
    'Parakhin Nikita',
    'Sherbakov Danil',
    'Sokolnikov Egor',
    'Makarov Alexander'
]
firstFrame = None
frame_number = 0
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    frame_number += 1
    if not ret:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        
        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
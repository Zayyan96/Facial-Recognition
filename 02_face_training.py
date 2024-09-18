import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = '/Users/zayyan/Downloads/FacialRecognitionProject/dataset/'

# Ensure the trainer directory exists
trainer_dir = '/Users/zayyan/Downloads/FacialRecognitionProject/trainer/'
if not os.path.exists(trainer_dir):
    os.makedirs(trainer_dir)

trainer_path = os.path.join(trainer_dir, 'trainer.yml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("/Users/zayyan/Downloads/FacialRecognitionProject/haarcascade_frontalface_default.xml")

# function to get the images and label data
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]     
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        try:
            PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
            img_numpy = np.array(PIL_img, 'uint8')

            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y+h, x:x+w])
                ids.append(id)
        except Exception as e:
            print(f"[ERROR] Could not process image {imagePath}: {e}")

    return faceSamples, ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.write(trainer_path)

# Print the number of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

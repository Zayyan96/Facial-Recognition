import cv2
import os

# Directory path for storing the dataset
dataset_dir = '/Users/zayyan/Downloads/FacialRecognitionProject/dataset/'

# Create the dataset directory if it doesn't exist
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier('/Users/zayyan/Downloads/FacialRecognitionProject/haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look at the camera and wait ...")
# Initialize individual sampling face count
count = 0

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1) # flip video image horizontally
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # Save the captured image into the datasets folder
        file_path = os.path.join(dataset_dir, "User." + str(face_id) + '.' + str(count) + ".jpg")
        print(f"[INFO] Saving image {file_path}")
        cv2.imwrite(file_path, gray[y:y + h, x:x + w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 100: # Take 30 face samples and stop video
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()

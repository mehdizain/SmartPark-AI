import cv2
import pickle
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog

path = "./drive/MyDrive/AI/ParkingAI/"

model = pickle.load(open(path + "model_vehicle.pkl", "rb"))

mask = cv2.imread(path + "mask_1920_1080.png", cv2.IMREAD_GRAYSCALE)
_, binary_mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

parking_spots = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    parking_spots.append([x, y, w, h])

vid = cv2.VideoCapture(path + "parking_1920_1080.mp4")

frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = vid.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(path + 'parking_output.avi',
                      cv2.VideoWriter_fourcc(*'XVID'),
                      fps,
                      (frame_width, frame_height))


frame_count = 0
skip_interval = int(fps * 5) #predict after 5 seconds, doing this for performance issues.
last_labels = [0] * len(parking_spots)

while True:
    ret, frame = vid.read()
    if not ret:
        break

    if frame_count % skip_interval == 0:
        current_labels = []
        for x, y, w, h in parking_spots:
            crop = frame[y:y+h, x:x+w]
            resized = resize(crop, (64, 64))
            gray = rgb2gray(resized)
            features = hog(gray)
            label = model.predict(features.reshape(1, -1))[0]
            current_labels.append(label)
        last_labels = current_labels

    for (x, y, w, h), label in zip(parking_spots, last_labels):
        color = (0, 255, 0) if label == 0 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    out.write(frame)
    frame_count += 1

vid.release()
out.release()

from ultralytics import YOLO
from PIL import Image
import cv2
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

WEIGHT_PATH = "../yolov5/runs/detect/train/weights/best.pt"
model = YOLO(WEIGHT_PATH)

hazy_img_path = "C:/Users/Administrator/Desktop/datasets/SOTs/Detection/foggy road scenes.v1i.yolov5pytorch/test/images/981c2476-41078bb6_jpg.rf.d648088811e38f06abc835a06b259270.jpg"
dehaze_img_path = "C:/Users/Administrator/Desktop/datasets/SOTs/Detection/foggy road scenes.v1i.yolov5pytorch/preds/images/981c2476-41078bb6_jpg.rf.d648088811e38f06abc835a06b259270.jpg"

hazy_img = cv2.imread(hazy_img_path)
dehaze_img = cv2.imread(dehaze_img_path)

results = model.predict(source=[hazy_img, dehaze_img], save=True, save_txt=True)  # save predictions as labels

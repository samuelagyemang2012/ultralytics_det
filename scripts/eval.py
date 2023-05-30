from ultralytics import YOLO
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == "__main__":
    WEIGHT_PATH = "../yolov8/runs/detect/train4/weights/best.pt"
    DATA_PATH = "../yamls/hazy.yaml"

    model = YOLO(WEIGHT_PATH)

    model.val(data=DATA_PATH)

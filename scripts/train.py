from ultralytics import YOLO
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    # Data path
    DATA_PATH = "../yamls/data.yaml"

    # Weights file
    WEIGHTS_PATH = "../runs/detect/train3/weights/last.pt"
    # resume_path = "yolov8/runs/detect/train3/weights/best.pt"

    # Parameters
    EPOCHS = 200
    IMG_SZE = 640
    BATCH = 16
    WORKERS = 0
    DEVICE = "0"

    # Model
    model = YOLO(WEIGHTS_PATH)

    # Train
    model.train(data=DATA_PATH, epochs=EPOCHS, imgsz=IMG_SZE, batch=BATCH, workers=WORKERS, device=DEVICE)

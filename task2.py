from imp import load_module
from ultralytics import YOLO
import cv2
import streamlit as st
from io import BytesIO
import numpy as np
# # import torch

# # print("Return the number of gpu with given index", torch.cuda.get_device_name(0))

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
model.train(data="config.yaml", epochs=10)  # train the model


# # Load the model from the specified path
# model_path = "C:/Users/Hussain Afroz/Desktop/PythonFiles/PythonTasks/runs/detect/train3/weights/best.pt"
# model = YOLO(model_path)

# # Function to predict


# def predict(model, img, classes=[], conf=0.5):
#     if classes:
#         res = model.predict(img, classes=classes, conf=conf)
#     else:
#         res = model.predict(img, conf=conf)
#     return res

# # Function to predict and detect objects


# def predict_and_detect(model, img, classes=[], conf=0.5):
#     img_copy = img.copy()
#     res = predict(model, img, classes, conf=conf)
#     for r in res:
#         for box in r.boxes:
#             cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
#                           (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
#             cv2.putText(img, f"{r.names[int(box.cls[0])]}",
#                         (int(box.xyxy[0][0]), int(box.xyxy[0][1])-10),
#                         cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
#     return img, res


# st.title("Object Detections")
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

# if uploaded_file is not None:
#     img_bytes = uploaded_file.getvalue()
#     orig_image = cv2.imdecode(np.frombuffer(
#         img_bytes, np.uint8), cv2.IMREAD_COLOR)
#     res_img, _ = predict_and_detect(model, orig_image, classes=[0], conf=0.5)

#     st.subheader("Original Image")
#     st.image(orig_image, caption='Original Image', use_column_width=True)

#     st.subheader("Detected Objects")
#     st.image(res_img, caption='Detected Objects', use_column_width=True)


from pathlib import Path
from PIL import Image
import streamlit as st
from ultralytics import YOLO
from utils import load_model, infer_uploaded_image, infer_uploaded_video

# setting page layout
st.set_page_config(
    page_title="Interactive Interface for YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# main page heading
st.title("Interactive Interface for YOLOv8")

# sidebar
st.sidebar.header("DL Model Config")

# model options
task_type = st.sidebar.selectbox(
    "Select Task",
    ["Detection"]
)

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 30, 100, 50)) / 100


model_path = "C:/Users/Hussain Afroz/Desktop/PythonFiles/PythonTasks/runs/detect/train3/weights/last.pt"

# if model_type:
#     model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
# else:
#     st.error("Please Select Model in Sidebar")

# # load pretrained DL model
#
model = YOLO(model_path)
# image/video options
st.sidebar.header("Image/Video Config")
source_selectbox = st.sidebar.selectbox(
    "Select Source",
    ["Image", "Video"]

)

source_img = None
if source_selectbox == "Image":  # Image
    infer_uploaded_image(confidence, model)
elif source_selectbox == "Video":  # Video
    infer_uploaded_video(confidence, model)
else:
    st.error("Currently only 'Image' and 'Video' source are implemented")

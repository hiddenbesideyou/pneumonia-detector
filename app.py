import streamlit as st
import torch
import cv2
import pydicom
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models import efficientnet_b4
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead

# ------------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------------
# Use CPU for the web app to save costs/compatibility (Streamlit Cloud has no GPU on free tier)
DEVICE = "cpu" 
CONFIG = {
    "img_size": 640,
    "cls_threshold": 0.50,
    "suspicious_threshold": 0.12, 
    "det_threshold": 0.20,
}

# ------------------------------------------------------------------
# 2. MODEL LOADING
# ------------------------------------------------------------------
import gdown
import os

@st.cache_resource
def load_models():
    # -----------------------------------------------------------
    # REPLACE THESE WITH YOUR ACTUAL FILE IDs FROM STEP 1
    # -----------------------------------------------------------
    CLS_ID = '1Ev2SIUYezx3dHKnddLIwYTu2Gdfjgmci' 
    DET_ID = '1xyfMylLza_iieeFxxij1gdZgrJy619Eg'
    # -----------------------------------------------------------
    
    CLS_PATH = "cls_fold0_best.pt"
    DET_PATH = "det_fold0_best.pt"

    # Download Classifier if missing
    if not os.path.exists(CLS_PATH):
        url = f'https://drive.google.com/uc?id={CLS_ID}'
        gdown.download(url, CLS_PATH, quiet=False)

    # Download Detector if missing
    if not os.path.exists(DET_PATH):
        url = f'https://drive.google.com/uc?id={DET_ID}'
        gdown.download(url, DET_PATH, quiet=False)

    # --- Standard Loading Logic Below ---
    if not (os.path.exists(CLS_PATH) and os.path.exists(DET_PATH)):
        st.error("❌ Failed to download models! Check your Google Drive IDs.")
        return None, None

    # Load Classifier
    cls_model = efficientnet_b4(weights=None)
    cls_model.classifier[1] = torch.nn.Linear(cls_model.classifier[1].in_features, 1)
    # Map to CPU
    cls_ckpt = torch.load(CLS_PATH, map_location=torch.device('cpu'), weights_only=False)
    cls_model.load_state_dict(cls_ckpt['model_state_dict'])
    cls_model.to(DEVICE)
    cls_model.eval()

    # Load Detector
    det_model = fasterrcnn_resnet50_fpn_v2(weights=None)
    anchor_generator = AnchorGenerator(sizes=((32,), (64,), (128,), (256,), (512,)), aspect_ratios=((0.5, 1.0, 2.0),) * 5)
    det_model.rpn.anchor_generator = anchor_generator
    det_model.rpn.head = RPNHead(det_model.backbone.out_channels, anchor_generator.num_anchors_per_location()[0])
    det_model.roi_heads.box_predictor = FastRCNNPredictor(det_model.roi_heads.box_predictor.cls_score.in_features, 2)
    
    det_ckpt = torch.load(DET_PATH, map_location=torch.device('cpu'), weights_only=False)
    det_model.load_state_dict(det_ckpt['model_state_dict'])
    det_model.to(DEVICE)
    det_model.eval()
    
    return cls_model, det_model

# ------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ------------------------------------------------------------------
def process_uploaded_file(uploaded_file):
    # Read file into bytes
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    if uploaded_file.name.lower().endswith('.dcm'):
        uploaded_file.seek(0)
        dcm = pydicom.dcmread(uploaded_file)
        img = dcm.pixel_array.astype(np.float32)
        # Normalize
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        # Standard Image
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    return img

def run_inference(img_rgb, cls_model, det_model):
    # Resize and Normalize
    transform = A.Compose([A.Resize(CONFIG['img_size'], CONFIG['img_size']), A.Normalize(), ToTensorV2()])
    aug = transform(image=img_rgb)
    img_tensor = aug['image'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        cls_prob = torch.sigmoid(cls_model(img_tensor)).item()
        det_out = det_model(img_tensor)[0]
    
    boxes = det_out['boxes'].cpu().numpy()
    scores = det_out['scores'].cpu().numpy()
    
    # Filter by threshold
    keep = scores > CONFIG['det_threshold']
    boxes = boxes[keep]
    scores = scores[keep]
    
    # Golden Mean Logic
    is_pneumonia = False
    logic_reason = "Normal"

    if cls_prob > CONFIG['cls_threshold']:
        is_pneumonia = True
        logic_reason = "High Confidence (Classifier)"
    elif cls_prob > CONFIG['suspicious_threshold'] and len(boxes) > 0:
        is_pneumonia = True
        logic_reason = "Safety Net (Suspicious + Boxes)"
    else:
        is_pneumonia = False
        logic_reason = "Normal"
        boxes = [] # Clear boxes if normal

    return is_pneumonia, cls_prob, boxes, scores, logic_reason

# ------------------------------------------------------------------
# 4. USER INTERFACE
# ------------------------------------------------------------------
st.set_page_config(page_title="AI Radiologist", page_icon="🩻", layout="wide")

st.title("🩻 AI Pneumonia Detection")
st.markdown("""
Upload one or multiple Chest X-Rays (**DICOM**, **JPG**, **PNG**) to detect Pneumonia.
*The model uses a 2-stage pipeline (EfficientNet + Faster R-CNN) to reduce false positives.*
""")

# Load Models
with st.spinner("Initializing AI Brain... (This may take a minute)"):
    cls_model, det_model = load_models()

if not cls_model:
    st.stop() # Stop execution if models failed

# File Uploader
# "accept_multiple_files=True" allows users to select a whole folder's contents (Ctrl+A)
uploaded_files = st.file_uploader("Drag and drop files here", type=['dcm', 'png', 'jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    st.success(f"Processing {len(uploaded_files)} images...")
    
    # Create a clean grid layout
    for uploaded_file in uploaded_files:
        with st.container():
            col1, col2 = st.columns([1, 2])
            
            try:
                img = process_uploaded_file(uploaded_file)
                is_pneumonia, prob, boxes, scores, reason = run_inference(img, cls_model, det_model)
                
                # Draw boxes for display
                vis_img = img.copy()
                vis_img = cv2.resize(vis_img, (640, 640)) # Resize for consistent display
                
                if len(boxes) > 0:
                    for box, score in zip(boxes, scores):
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                        cv2.putText(vis_img, f"{score:.0%}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Left Column: Image
                with col1:
                    st.image(vis_img, use_container_width=True)
                
                # Right Column: Diagnosis
                with col2:
                    st.subheader(f"📄 {uploaded_file.name}")
                    
                    if is_pneumonia:
                        st.error(f"**Diagnosis: PNEUMONIA DETECTED**")
                    else:
                        st.success(f"**Diagnosis: NORMAL**")
                        
                    st.write(f"**Probability:** {prob:.1%}")
                    st.info(f"**AI Logic:** {reason}")
                    
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")
            
            st.divider()
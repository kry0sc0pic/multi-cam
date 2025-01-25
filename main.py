# Imports
import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from annotated_text import annotated_text

# CAM Algorithms
from pytorch_grad_cam import GradCAM, FullGrad, HiResCAM, LayerCAM

# Metrics
from pytorch_grad_cam.metrics.road import ROADMostRelevantFirst

# Utils
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# Preprocess Transform
transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Streamlit UI Conf
st.set_page_config(
    page_title="Multi-Model Class Activation Maps for Explainable Predictions",
    page_icon="🔨"
)

st.title("Multi-Model Class Activation Maps for Explainable Predictions")
st.subheader("by Krishaay Jois `23MT7013` | `DY23ENGP0MBT017`")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"],accept_multiple_files=False)
labels = {idx: label.strip() for idx, label in enumerate(open('classes.txt'))} 

CAMS = []
CAM_NAMES = [
    "GradCAM",
    "FullGrad",
    "HiResCAM",
    "LayerCAM"
]
CAM_MAPS = []
VIZ = []
SCORES = []


if uploaded_file is not None:
    # Load Selected Image
    image = Image.open(uploaded_file)
    input_tensor = transforms(image)
    input_batch = input_tensor.unsqueeze(0)
    base_rgb_image = np.array(image.resize((224, 224))) / 255

    # Load Resnet
    model = models.resnet50(pretrained=True)
    model.eval()

    # Inference
    with torch.no_grad():
        output = model(input_batch)

    predicted_class = output.argmax().item()
    targets = [ClassifierOutputTarget(predicted_class)]
    target_layers = [model.layer4[-1]]

    # Create cam objects
    grad_cam = GradCAM(model=model, target_layers=target_layers)
    full_grad = FullGrad(model=model, target_layers=target_layers)
    hirescam = HiResCAM(model=model, target_layers=target_layers)
    layer_cam = LayerCAM(model=model, target_layers=target_layers)

    CAMS.append(grad_cam)
    CAMS.append(full_grad)
    CAMS.append(hirescam)
    CAMS.append(layer_cam)

    # Create metrics
    road_mr = ROADMostRelevantFirst()

    for cam in CAMS:
        # get CAM
        grayscale_cam = cam(input_tensor=input_batch,
                            aug_smooth=True,
                            eigen_smooth=True,
                           targets=targets)

        # Add map
        CAM_MAPS.append(grayscale_cam[0])


    
        scores, _ = road_mr(input_batch, grayscale_cam,targets,model,True)
        score = scores[0]
        SCORES.append((score)) # lower scores are    better
       
        # Add visual
        visual = show_cam_on_image(base_rgb_image, grayscale_cam[0], use_rgb=True)

        VIZ.append(visual)


    ## Display
    st.subheader("Class: `" + labels[predicted_class].replace('_',' ').capitalize()+ "`")

    st.subheader("Scored Explanation Maps")
    st.text("Lower scores are better")
    cols = st.columns(len(CAM_NAMES))
    for i in range(len(CAM_NAMES)):
        cols[i].subheader(CAM_NAMES[i])
        # score upto 2 decimals
        cols[i].write(f"#### Score: `{SCORES[i]:.2f}`")
        cols[i].image(VIZ[i], use_container_width=False)
    
    offset_scores = [abs(score - max(SCORES)) for score in SCORES]
    norm_scores = [score/sum(offset_scores) for score in offset_scores]
    
    weighted_cam = np.average(CAM_MAPS, axis=0, weights=norm_scores)
    st.subheader("Weighted Explanation Map")
    for i in range(len(CAM_NAMES)):
        st.write(f"{CAM_NAMES[i]}: `{norm_scores[i]:.2f}`")
    st.image(show_cam_on_image(base_rgb_image, weighted_cam, use_rgb=True), use_container_width=False)
 

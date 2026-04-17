import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

from reportlab.platypus import Image as RLImage
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from datetime import datetime
import tempfile

# -------- PAGE CONFIG --------
st.set_page_config(page_title="DR Screening", layout="wide")

st.markdown("""
<style>
body {
    background-color: #f5f7fb;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# -------- DEVICE --------
device = torch.device("cpu")

# -------- MODEL --------
model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 5)

model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# -------- TRANSFORM --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------- PREDICTION --------
def predict_image(image):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    return pred.item(), confidence.item(), image

# -------- REPORT --------
def generate_report(pred_class, confidence, language, name, age):

    grades_en = {
        0: "No DR",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        4: "Proliferative"
    }

    grades_ta = {
        0: "நோய் இல்லை",
        1: "லேசான நிலை",
        2: "மிதமான நிலை",
        3: "கடுமையான நிலை",
        4: "மிகவும் கடுமையான நிலை"
    }

    rec_en = {
        0: "Routine checkup",
        1: "6-12 months follow-up",
        2: "Consult doctor",
        3: "Urgent consultation",
        4: "Immediate treatment"
    }

    rec_ta = {
        0: "சாதாரண பரிசோதனை",
        1: "6-12 மாதங்களில் பரிசோதனை",
        2: "மருத்துவரை அணுகவும்",
        3: "உடனடி ஆலோசனை",
        4: "உடனடி சிகிச்சை"
    }

    if language == "Tamil":
        grade = grades_ta[pred_class]
        rec = rec_ta[pred_class]
    else:
        grade = grades_en[pred_class]
        rec = rec_en[pred_class]

    # UI Display
    st.markdown(f"""
    ### 👤 Patient Info
    - **Name:** {name}
    - **Age:** {age}

    ### 🧠 Diagnosis
    - **Stage:** {grade}
    - **Confidence:** {confidence*100:.2f}%

    ### 📋 Recommendation
    {rec}
    """)

    st.progress(int(confidence * 100))

    # Downloadable report
    report = f"""
--- Diabetic Retinopathy Report ---

Patient Name : {name}
Age          : {age}

Diagnosis    : {grade}
Confidence   : {confidence*100:.2f} %

Recommendation:
{rec}

-----------------------------------
"""
    return report, grade, rec

def create_advanced_pdf(name, age, grade, confidence, rec, overlay_img):

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()
    content = []

    # -------- HEADER --------
    content.append(Paragraph("🏥 Diabetic Retinopathy Screening Report", styles['Title']))
    content.append(Spacer(1, 10))

    # -------- DATE --------
    date_str = datetime.now().strftime("%d-%m-%Y %H:%M")
    content.append(Paragraph(f"Report Generated: {date_str}", styles['Normal']))
    content.append(Spacer(1, 15))

    # -------- PATIENT TABLE --------
    patient_data = [
        ["Patient Name", name],
        ["Age", str(age)],
        ["Diagnosis", grade],
        ["Confidence", f"{confidence*100:.2f}%"]
    ]

    table = Table(patient_data, colWidths=[150, 250])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
    ]))

    content.append(table)
    content.append(Spacer(1, 20))

    # -------- RECOMMENDATION --------
    content.append(Paragraph("Clinical Recommendation:", styles['Heading2']))
    content.append(Paragraph(rec, styles['Normal']))
    content.append(Spacer(1, 20))

    # -------- IMAGE (Grad-CAM Overlay) --------
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        cv2.imwrite(temp_file.name, overlay_img)

        content.append(Paragraph("Model Attention (Grad-CAM):", styles['Heading2']))
        content.append(Spacer(1, 10))

        img = RLImage(temp_file.name, width=300, height=300)
        content.append(img)
        content.append(Spacer(1, 20))

    except Exception as e:
        content.append(Paragraph("Image could not be loaded", styles['Normal']))

    # -------- DISCLAIMER --------
    content.append(Paragraph(
        "⚠️ This is an AI-assisted screening result. Not a final medical diagnosis. Please consult an ophthalmologist.",
        styles['Italic']
    ))

    doc.build(content)

    buffer.seek(0)
    return buffer


# -------- GRADCAM --------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_image, target_class):
        self.model.zero_grad()
        output = self.model(input_image)

        loss = output[:, target_class]
        loss.backward()

        gradients = self.gradients
        activations = self.activations

        pooled_gradients = torch.mean(gradients, dim=[0,2,3])

        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        heatmap /= torch.max(heatmap)

        return heatmap.cpu().detach().numpy()

target_layer = model.features[-1][0]
gradcam = GradCAM(model, target_layer)

# -------- UI --------
st.title("🩺 Diabetic Retinopathy Screening System")

# Sidebar
st.sidebar.title("Patient Details")
name = st.sidebar.text_input("Patient Name")
age = st.sidebar.number_input("Age", 1, 120)
language = st.sidebar.selectbox("Select Language", ["English", "Tamil"])

uploaded_file = st.file_uploader("Upload Retinal Image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    if not name:
        st.warning("Please enter patient name")
        st.stop()

    image = Image.open(uploaded_file).convert("RGB")

    pred, conf, tensor = predict_image(image)

    col1, col2 = st.columns(2)

    # LEFT - IMAGE
    with col1:
        st.image(image, caption="Retinal Image", use_column_width=True)

    # RIGHT - REPORT
    with col2:
        st.markdown("## Diagnosis Result")
        report, grade, rec = generate_report(pred, conf, language, name, age)


    # -------- GRADCAM OVERLAY --------
    heatmap = gradcam.generate(tensor, pred)

    img = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)

    st.markdown("## 🔍 Model Attention (Grad-CAM)")
    st.image(overlay, caption="Highlighted Disease Regions")

    pdf = create_advanced_pdf(name, age, grade, conf, rec, overlay)

    st.download_button(
            label="📄 Download Full Medical Report",
            data=pdf,
            file_name="DR_Advanced_Report.pdf",
            mime="application/pdf"
        )

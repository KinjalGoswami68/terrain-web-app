import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

# --- 1. CINEMATIC UI SETUP ---
st.set_page_config(page_title="Geological AI", page_icon="🛰️", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Inter:wght@300;400&display=swap');
    
    .stApp { background: radial-gradient(circle at 50% 10%, #0a1a12 0%, #020503 100%); }
    
    .block-container {
        background: rgba(10, 25, 16, 0.4);
        border: 1px solid rgba(80, 200, 120, 0.15);
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(12px);
        padding: 3rem !important;
        margin-top: 2rem;
    }

    h1 {
        font-family: 'Orbitron', sans-serif !important;
        color: #50C878 !important;
        text-shadow: 0 0 10px rgba(80, 200, 120, 0.4), 0 0 20px rgba(80, 200, 120, 0.2);
        text-align: center;
        letter-spacing: 2px;
    }
    h3 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 300 !important;
        color: #8fa898 !important;
        text-align: center;
        letter-spacing: 1px;
        margin-bottom: 2rem !important;
    }

    .stButton>button {
        background: linear-gradient(90deg, #030a06 0%, #0a1f12 100%);
        color: #50C878;
        border: 1px solid #50C878;
        font-family: 'Orbitron', sans-serif !important;
        font-size: 1.2rem;
        letter-spacing: 2px;
        width: 100%;
        padding: 15px;
        transition: 0.4s;
    }
    .stButton>button:hover {
        background: #50C878;
        color: #000;
        box-shadow: 0 0 20px rgba(80, 200, 120, 0.8);
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🛰️ ORBITAL TERRAIN SCANNER</h1>", unsafe_allow_html=True)
st.markdown("<h3>AI-DRIVEN GEOLOGICAL EXPLORATION & RADIOMETRIC ANALYSIS</h3>", unsafe_allow_html=True)

# --- NEW UPLOAD SYSTEM ---
uploaded_file = st.sidebar.file_uploader("📡 Upload Custom Satellite Scan", type=["tif", "jpg", "png"])

if uploaded_file is not None:
    # If the user uploads a file, save it temporarily so the AI can read it
    with open("temp_scan.tif", "wb") as f:
        f.write(uploaded_file.getbuffer())
    target_scan = "temp_scan.tif"
else:
    # If no file is uploaded, default to your built-in static demo
    target_scan = "map.tif"
# -------------------------

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
               'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
    
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load('finetuned.pth', map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model, device, classes

model, device, classes = load_model()

# --- 3. SESSION STATE INIT ---
if 'scan_complete' not in st.session_state:
    st.session_state.scan_complete = False
    st.session_state.map_image = None
    st.session_state.results = None

# --- 4. MAIN SCANNING LOGIC ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("INITIATE SATELLITE UPLINK"):
        with st.spinner("Decrypting optical data and mapping topography..."):
            time.sleep(1.5) 
            
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            try:
                # Updated to use the dynamic target_scan variable
                with rasterio.open(target_scan) as dataset:
                    r, g, b = dataset.read(1), dataset.read(2), dataset.read(3)
                    map_image = np.dstack((r, g, b))

                H, W, _ = map_image.shape
                patch_size = 64
                grid_h, grid_w = H // patch_size, W // patch_size
                results = np.zeros((grid_h, grid_w))

                with torch.no_grad():
                    for y in range(grid_h):
                        for x in range(grid_w):
                            y_start, x_start = y * patch_size, x * patch_size
                            patch = map_image[y_start:y_start+patch_size, x_start:x_start+patch_size]
                            
                            img_pil = Image.fromarray(patch)
                            input_tensor = transform(img_pil).unsqueeze(0).to(device)
                            
                            outputs = model(input_tensor)
                            _, predicted = torch.max(outputs.data, 1)
                            results[y, x] = predicted.item()
                
                # Save to session state so we don't have to rescan
                st.session_state.map_image = map_image
                st.session_state.results = results
                st.session_state.scan_complete = True

            except Exception as e:
                st.error(f"Critical System Failure: {e}")

# --- 5. RENDER RESULTS & ANOMALY FILTER ---
if st.session_state.scan_complete:
    st.divider()
    
    # THE ANOMALY FILTER TOGGLE
    st.markdown("<h4 style='color: #50C878; text-align: center; font-family: Orbitron;'>TARGETING SYSTEMS</h4>", unsafe_allow_html=True)
    col_t1, col_t2, col_t3 = st.columns([1, 1.5, 1])
    with col_t2:
        filter_enabled = st.toggle("🎯 ISOLATE EXPOSED GROUND (HIDE VEGETATION & WATER)")

    # Apply masking logic if toggle is flipped
    display_map = st.session_state.map_image.copy()
    display_results = st.session_state.results.copy()
    
    if filter_enabled:
        # Classes to hide: 0(Crop), 1(Forest), 5(Pasture), 6(PermCrop), 8(River), 9(SeaLake)
        # We are looking for exposed land, highways, industrial areas, etc.
        hide_classes = [0, 1, 5, 6, 8, 9] 
        mask = np.isin(display_results, hide_classes)
        
        # Black out those areas on the AI classification map (-1 sets it below color map limits)
        display_results[mask] = -1 
        
        # Dim the raw map where there is vegetation/water to highlight exposed ground
        # Because grid is 64x64 patches, we expand the mask to match the raw image size
        for y in range(display_results.shape[0]):
            for x in range(display_results.shape[1]):
                if display_results[y, x] == -1:
                    y_start, x_start = y * 64, x * 64
                    display_map[y_start:y_start+64, x_start:x_start+64] = display_map[y_start:y_start+64, x_start:x_start+64] // 4

    # Graph Rendering
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('none')
    
    ax1.imshow(display_map)
    title1 = "RAW FEED (FILTERED)" if filter_enabled else "RAW OPTICAL FEED"
    ax1.set_title(title1, color='#50C878', fontname='Courier New', fontsize=14, pad=15)
    ax1.axis('off')

    cmap = plt.get_cmap('magma') 
    # Set background color for masked (-1) areas to black
    cmap.set_under('black') 
    
    im = ax2.imshow(display_results, cmap=cmap, vmin=0, vmax=9)
    title2 = "EXPOSED GROUND ISOLATED" if filter_enabled else "AI CLASSIFICATION MATRIX"
    ax2.set_title(title2, color='#50C878', fontname='Courier New', fontsize=14, pad=15)
    ax2.axis('off')

    cbar = fig.colorbar(im, ax=ax2, ticks=range(10), fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(classes, fontname='Courier New', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='#50C878', labelcolor='#8fa898') 

    st.pyplot(fig)

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from skimage.transform import radon, iradon
from skimage.data import shepp_logan_phantom

# 1. Page Configuration & Futuristic Styling
st.set_page_config(page_title="MedMath 3D Analytics", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0B0F19; color: #E2E8F0; }
    h1, h2, h3 { color: #00F2FE !important; font-family: 'Trebuchet MS', sans-serif; }
    .stSelectbox, .stSlider { background-color: #111827; border-radius: 10px; padding: 10px; }
    div[data-baseweb="tab-list"] { gap: 8px; }
    button[data-baseweb="tab"] {
        background-color: #1E293B;
        border-radius: 8px;
        color: #94A3B8;
        padding: 8px 16px;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #00F2FE !important;
        color: #090D16 !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ MedMath Analytics 3D Platform")
st.caption("Advanced Mathematical Imaging & Computational Physics Lab")

tab1, tab2, tab3 = st.tabs([
    "🖥️ 3D CT Tomography", 
    "🧠 MRI 3D k-Space Fourier Lab", 
    "📡 3D Ultrasound Wave Simulator"
])

# ==========================================
# TAB 1: 3D CT TOMOGRAPHY SIMULATOR
# ==========================================
with tab1:
    st.header("CT Scan: 3D Density Map & Radon Reconstructions")
    col1, col2 = st.columns([1, 2.5])

    with col1:
        num_angles = st.slider("X-Ray Projection Angles:", min_value=10, max_value=180, value=90, step=5)
        st.latex(r"Ax = b")
        st.latex(r"R_\theta(x_1, x_2) = \int_{-\infty}^{\infty} f(x_1 \cos\theta + t \sin\theta, x_1 \sin\theta - t \cos\theta) \, dt")
        st.info("💡 **3D Projection Math:** High-resolution slice values are rendered in 3D topography to analyze tissue density gradients.")

    with col2:
        image = shepp_logan_phantom()
        theta = np.linspace(0.0, 180.0, num_angles, endpoint=False)
        sinogram = radon(image, theta=theta)
        reconstructed = iradon(sinogram, theta=theta, filter_name='ramp')

        # 3D Surface Plot of Reconstructed Phantom
        x_mesh, y_mesh = np.meshgrid(np.arange(reconstructed.shape[1]), np.arange(reconstructed.shape[0]))
        fig3d_ct = go.Figure(data=[go.Surface(
            z=reconstructed, x=x_mesh, y=y_mesh, 
            colorscale='Electric', showscale=False
        )])
        fig3d_ct.update_layout(
            title="3D Reconstructed Density Terrain (Rotate/Zoom)",
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(title="Tissue Density"),
                aspectratio=dict(x=1, y=1, z=0.4),
                bgcolor='#0B0F19'
            ),
            paper_bgcolor='#0B0F19',
            font=dict(color='white'),
            margin=dict(l=0, r=0, b=0, t=40),
            height=500
        )
        st.plotly_chart(fig3d_ct, use_container_width=True)

# ==========================================
# TAB 2: MRI 3D K-SPACE LAB
# ==========================================
with tab2:
    st.header("MRI: 3D k-Space Frequency Spectrum & 2D IFFT Reconstruction")
    col1, col2 = st.columns([1, 2.5])

    with col1:
        filter_mode = st.radio("k-Space Frequency Filter:", [
            "Full Spectrum (Complete)", 
            "Low-Pass (Soft Tissue Details)", 
            "High-Pass (Edge Detection)"
        ])
        st.latex(r"F(u,v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x,y) e^{-j 2\pi (\frac{ux}{M} + \frac{vy}{N})}")
        st.info("💡 **Frequency Geometry:** The central spike stores macroscopic brightness, while the surrounding 3D matrix represents high-frequency structural edges.")

    with col2:
        img_mri = shepp_logan_phantom()
        f_transform = np.fft.fftshift(np.fft.fft2(img_mri))
        rows, cols = img_mri.shape
        crow, ccol = rows // 2, cols // 2

        mask = np.ones((rows, cols), np.uint8)
        if filter_mode == "Low-Pass (Soft Tissue Details)":
            r = 25
            y, x = np.ogrid[:rows, :cols]
            mask_area = (x - ccol)**2 + (y - crow)**2 > r**2
            mask[mask_area] = 0
        elif filter_mode == "High-Pass (Edge Detection)":
            r = 15
            y, x = np.ogrid[:rows, :cols]
            mask_area = (x - ccol)**2 + (y - crow)**2 <= r**2
            mask[mask_area] = 0

        f_transform_filtered = f_transform * mask
        k_space_visual = np.log(np.abs(f_transform_filtered) + 1)

        # 3D k-space Surface Mesh
        x_m, y_m = np.meshgrid(np.arange(cols), np.arange(rows))
        fig3d_mri = go.Figure(data=[go.Surface(
            z=k_space_visual, x=x_m, y=y_m, 
            colorscale='Viridis', showscale=False
        )])
        fig3d_mri.update_layout(
            title="3D $k$-Space Frequency Matrix Topology",
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(title="Log Magnitude"),
                aspectratio=dict(x=1, y=1, z=0.4),
                bgcolor='#0B0F19'
            ),
            paper_bgcolor='#0B0F19',
            font=dict(color='white'),
            margin=dict(l=0, r=0, b=0, t=40),
            height=500
        )
        st.plotly_chart(fig3d_mri, use_container_width=True)

# ==========================================
# TAB 3: 3D ULTRASOUND WAVE SIMULATOR
# ==========================================
with tab3:
    st.header("Ultrasound: Interactive 3D Soundwave Propagation")
    col1, col2 = st.columns([1, 2.5])

    with col1:
        tissue_choice = st.selectbox("Interface Boundary:", ["Fat to Muscle", "Muscle to Bone", "Fat to Bone"])
        freq = st.slider("Probe Transducer Frequency (MHz):", min_value=1.0, max_value=10.0, value=4.0, step=0.5)
        st.latex(r"d = \frac{v \cdot t}{2}")
        st.latex(r"R = \left(\frac{Z_2 - Z_1}{Z_2 + Z_1}\right)^2")

        z_dict = {"Fat": 1.38, "Muscle": 1.70, "Bone": 6.00}
        t1, t2 = tissue_choice.split(" to ")
        z1, z2 = z_dict[t1], z_dict[t2]
        r_coeff = ((z2 - z1) / (z2 + z1)) ** 2
        st.metric(label="Reflection Coefficient (R)", value=f"{r_coeff:.4f}")

    with col2:
        time_t = np.linspace(0, 10, 200)
        depth_y = np.linspace(0, 5, 50)
        T, Y = np.meshgrid(time_t, depth_y)

        # Modeling 3D Wave dissipation across depth and time
        wave_z = np.sin(2 * np.pi * freq * (T - Y/1.5)) * np.exp(-0.3 * Y) * np.exp(-0.2 * (T - 3)**2)
        echo_z = r_coeff * np.sin(2 * np.pi * freq * (T - Y/1.5)) * np.exp(-0.3 * Y) * np.exp(-0.2 * (T - 7)**2)
        total_wave = wave_z + echo_z

        fig3d_us = go.Figure(data=[go.Surface(
            z=total_wave, x=T, y=Y, 
            colorscale='Plasma', showscale=False
        )])
        fig3d_us.update_layout(
            title="3D Acoustic Pulse Echo Surface (Time vs Depth vs Amplitude)",
            scene=dict(
                xaxis=dict(title="Time (µs)"),
                yaxis=dict(title="Depth (cm)"),
                zaxis=dict(title="Amplitude"),
                aspectratio=dict(x=1.2, y=1, z=0.4),
                bgcolor='#0B0F19'
            ),
            paper_bgcolor='#0B0F19',
            font=dict(color='white'),
            margin=dict(l=0, r=0, b=0, t=40),
            height=500
        )
        st.plotly_chart(fig3d_us, use_container_width=True)

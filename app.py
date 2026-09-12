import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from skimage.transform import radon, iradon
from skimage.data import shepp_logan_phantom

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MedMath Analytics 3D Platform",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0A0E17;
        color: #E2E8F0;
    }
    h1, h2, h3 {
        color: #00F0FF !important;
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(0, 240, 255, 0.2);
        border-radius: 12px;
        padding: 18px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0F172A;
        padding: 8px;
        border-radius: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00F0FF !important;
        color: #0A0E17 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ MedMath Analytics 3D Platform")
st.caption("Advanced Mathematical Imaging & Computational Physics Lab")

# Base phantom data caching
@st.cache_data
def load_phantom():
    return shepp_logan_phantom()

base_phantom = load_phantom()

# -----------------------------------------------------------------------------
# 2. NAVIGATION TABS SETUP
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🖥️ 3D CT Tomography", 
    "🧠 MRI 3D k-Space Fourier Lab", 
    "🔊 3D Ultrasound Wave Simulator"
])

# =============================================================================
# TAB 1: CT SCAN SIMULATOR & LIVE X-RAY SLICES
# =============================================================================
with tab1:
    st.header("CT Scan: 3D Density Map & Radon Reconstructions")
    st.write("Simulate X-ray beam projections across angles to reconstruct spatial density matrices.")

    col1, col2 = st.columns([1.1, 2])

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        num_angles = st.slider("X-Ray Projection Angles:", min_value=4, max_value=180, value=90, step=2)
        
        st.latex(r"Ax = b")
        st.latex(r"R_\theta(x_1, x_2) = \int_{-\infty}^{\infty} f(x_1 \cos\theta + t \sin\theta, x_1 \sin\theta - t \cos\theta) \, dt")
        
        st.info("💡 **3D Projection Math:** High-resolution slice values are rendered in 3D topography to analyze tissue density gradients.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Calculate Radon transform and reconstruction
        theta = np.linspace(0.0, 180.0, num_angles, endpoint=False)
        sinogram = radon(base_phantom, theta=theta)
        reconstructed = iradon(sinogram, theta=theta, filter_name='ramp')

        # Matplotlib 2D Slice Figures
        fig_xray, ax_xray = plt.subplots(1, 2, figsize=(8, 2.8))
        fig_xray.patch.set_facecolor('#0A0E17')

        for a in ax_xray:
            a.set_facecolor('#0A0E17')
            a.title.set_color('#00F0FF')
            a.tick_params(colors='white')

        ax_xray[0].imshow(reconstructed, cmap='bone')
        ax_xray[0].set_title(f"Reconstructed Slice ({num_angles}°)")
        ax_xray[0].axis('off')

        ax_xray[1].imshow(sinogram, cmap='inferno', aspect='auto')
        ax_xray[1].set_title("Live Sinogram Matrix")
        ax_xray[1].axis('off')

        plt.tight_layout()
        st.pyplot(fig_xray)

        # 3D Density Terrain Surface Plotly
        sub_phantom = reconstructed[::2, ::2]
        fig_3d = go.Figure(data=[go.Surface(z=sub_phantom, colorscale='Plasma')])
        fig_3d.update_layout(
            title="3D Reconstructed Density Terrain (Rotate/Zoom)",
            title_font_color="#00F0FF",
            autosize=True,
            height=320,
            margin=dict(l=0, r=0, b=0, t=30),
            paper_bgcolor='#0A0E17',
            scene=dict(
                bgcolor='#0A0E17',
                zaxis_title='Tissue Density',
                xaxis_title='X Axis',
                yaxis_title='Y Axis'
            )
        )
        st.plotly_chart(fig_3d, use_container_width=True)

# =============================================================================
# TAB 2: MRI 3D k-SPACE LAB
# =============================================================================
with tab2:
    st.header("MRI: 3D k-Space Frequency Spectrum & 2D IFFT Reconstruction")

    col1, col2 = st.columns([1.1, 2])

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        filter_mode = st.radio("k-Space Frequency Filter:", ["Full Spectrum (Complete)", "Low-Pass (Soft Tissue Details)", "High-Pass (Edge Detection)"])
        st.latex(r"F(u,v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x,y) e^{-j 2\pi (\frac{ux}{M} + \frac{vy}{N})}")
        st.info("💡 **Frequency Geometry:** The central spike stores macroscopic brightness, while the surrounding 3D matrix represents high-frequency structural edges.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        f_transform = np.fft.fftshift(np.fft.fft2(base_phantom))
        rows, cols = base_phantom.shape
        crow, ccol = rows // 2, cols // 2

        mask = np.ones((rows, cols), np.uint8)
        if filter_mode == "Low-Pass (Soft Tissue Details)":
            r = 18
            y, x = np.ogrid[:rows, :cols]
            mask_area = (x - ccol)**2 + (y - crow)**2 > r**2
            mask[mask_area] = 0
        elif filter_mode == "High-Pass (Edge Detection)":
            r = 12
            y, x = np.ogrid[:rows, :cols]
            mask_area = (x - ccol)**2 + (y - crow)**2 <= r**2
            mask[mask_area] = 0

        f_transform_filtered = f_transform * mask
        k_space_visual = np.log(np.abs(f_transform_filtered) + 1)

        sub_kspace = k_space_visual[::4, ::4]
        fig_mesh = go.Figure(data=[go.Surface(z=sub_kspace, colorscale='Viridis')])
        fig_mesh.update_layout(
            title="3D k-Space Frequency Matrix Topology",
            title_font_color="#00F0FF",
            autosize=True,
            height=380,
            margin=dict(l=0, r=0, b=0, t=30),
            paper_bgcolor='#0A0E17',
            scene=dict(
                bgcolor='#0A0E17',
                zaxis_title='Log Magnitude'
            )
        )
        st.plotly_chart(fig_mesh, use_container_width=True)

# =============================================================================
# TAB 3: ULTRASOUND 3D WAVE SIMULATOR
# =============================================================================
with tab3:
    st.header("Ultrasound: Interactive 3D Soundwave Propagation")

    col1, col2 = st.columns([1.1, 2])

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        tissue_choice = st.selectbox("Interface Boundary:", ["Fat to Muscle", "Muscle to Bone", "Fat to Bone"])
        frequency = st.slider("Probe Transducer Frequency (MHz):", min_value=1.0, max_value=10.0, value=4.0, step=0.5)

        st.latex(r"d = \frac{v \cdot t}{2}")
        st.latex(r"R = \left(\frac{Z_2 - Z_1}{Z_2 + Z_1}\right)^2")

        z_dict = {"Fat": 1.38, "Muscle": 1.70, "Bone": 6.00}
        t1, t2 = tissue_choice.split(" to ")
        z1, z2 = z_dict[t1], z_dict[t2]
        r_coeff = ((z2 - z1) / (z2 + z1)) ** 2
        st.metric(label="Reflection Coefficient (R)", value=f"{r_coeff:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        t_time = np.linspace(0, 5, 100)
        depth = np.linspace(0, 3, 100)
        T, D = np.meshgrid(t_time, depth)

        Z_wave = np.sin(2 * np.pi * frequency * (T - D/1.54)) * np.exp(-0.8 * (T - D/1.54)**2)

        fig_us = go.Figure(data=[go.Surface(z=Z_wave, x=T, y=D, colorscale='Plasma')])
        fig_us.update_layout(
            title="3D Acoustic Pulse Echo Surface (Time vs Depth vs Amplitude)",
            title_font_color="#00F0FF",
            autosize=True,
            height=380,
            margin=dict(l=0, r=0, b=0, t=30),
            paper_bgcolor='#0A0E17',
            scene=dict(
                bgcolor='#0A0E17',
                xaxis_title='Time (µs)',
                yaxis_title='Depth',
                zaxis_title='Amplitude'
            )
        )
        st.plotly_chart(fig_us, use_container_width=True)

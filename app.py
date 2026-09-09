import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
from skimage.data import shepp_logan_phantom

# 1. Page Configuration
st.set_page_config(page_title="MedMath Analytics", layout="wide")

# Custom Styling
st.markdown("""
<style>
    h1, h2, h3 { color: #38BDF8; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ MedMath Analytics Platform")
st.subheader("Mathematical Principles of Modern Medical Imaging")

# 2. Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "1. CT Scan (Radon Transform)", 
    "2. MRI (2D Fourier Transform)", 
    "3. Ultrasound (Pulse-Echo)"
])

# ==========================================
# TAB 1: CT SCAN SIMULATOR
# ==========================================
with tab1:
    st.header("CT Scan: Linear Algebra & Filtered Back Projection")
    st.write("Computed Tomography uses X-ray projections across angles to reconstruct 2D tissue slices.")

    col1, col2 = st.columns([1, 2])

    with col1:
        num_angles = st.slider("Number of X-Ray Projection Angles:", min_value=4, max_value=180, value=136, step=2)
        st.latex(r"Ax = b")
        st.latex(r"R_\theta(x_1, x_2) = \int_{-\infty}^{\infty} f(x_1 \cos\theta + t \sin\theta, x_1 \sin\theta - t \cos\theta) \, dt")
        st.info("💡 **Key Math:** Few angles produce blurry line artifacts. Increasing angles solves the linear system accurately.")

    with col2:
        image = shepp_logan_phantom()
        theta = np.linspace(0.0, 180.0, num_angles, endpoint=False)
        sinogram = radon(image, theta=theta)
        reconstructed = iradon(sinogram, theta=theta, filter_name='ramp')

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        fig.patch.set_facecolor('#0E1117')

        for a in ax:
            a.set_facecolor('#0E1117')
            a.title.set_color('white')

        ax[0].imshow(image, cmap='gray')
        ax[0].set_title("Original Brain Slice")
        ax[0].axis('off')

        ax[1].imshow(sinogram, cmap='gray', aspect='auto')
        ax[1].set_title("Live Sinogram Matrix")
        ax[1].axis('off')

        ax[2].imshow(reconstructed, cmap='gray')
        ax[2].set_title(f"Reconstruction ({num_angles} Angles)")
        ax[2].axis('off')

        plt.tight_layout()
        st.pyplot(fig)

# ==========================================
# TAB 2: MRI FREQUENCY LAB
# ==========================================
with tab2:
    st.header("MRI: 2D Discrete Fourier Transform & k-Space")
    st.write("MRI scanners measure spatial frequency data (k-space) and reconstruct the image using 2D Inverse FFT.")

    col1, col2 = st.columns([1, 2])

    with col1:
        filter_mode = st.radio("k-Space Frequency Filter:", ["Full Spectrum (Complete)", "Low-Pass (Soft Details Only)", "High-Pass (Edges Only)"])
        st.latex(r"F(u,v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x,y) e^{-j 2\pi (\frac{ux}{M} + \frac{vy}{N})}")
        st.info("💡 **Key Math:** Center of k-space stores contrast/brightness; outer region stores edge sharp details.")

    with col2:
        img_mri = shepp_logan_phantom()
        f_transform = np.fft.fftshift(np.fft.fft2(img_mri))
        rows, cols = img_mri.shape
        crow, ccol = rows // 2, cols // 2

        mask = np.ones((rows, cols), np.uint8)
        if filter_mode == "Low-Pass (Soft Details Only)":
            r = 20
            y, x = np.ogrid[:rows, :cols]
            mask_area = (x - ccol)**2 + (y - crow)**2 > r**2
            mask[mask_area] = 0
        elif filter_mode == "High-Pass (Edges Only)":
            r = 15
            y, x = np.ogrid[:rows, :cols]
            mask_area = (x - ccol)**2 + (y - crow)**2 <= r**2
            mask[mask_area] = 0

        f_transform_filtered = f_transform * mask
        k_space_visual = np.log(np.abs(f_transform_filtered) + 1)
        img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(f_transform_filtered)))

        fig2, ax2 = plt.subplots(1, 2, figsize=(10, 4))
        fig2.patch.set_facecolor('#0E1117')

        for a in ax2:
            a.set_facecolor('#0E1117')
            a.title.set_color('white')

        ax2[0].imshow(k_space_visual, cmap='gray')
        ax2[0].set_title("Filtered k-Space Matrix")
        ax2[0].axis('off')

        ax2[1].imshow(img_back, cmap='gray')
        ax2[1].set_title("2D Inverse FFT Output")
        ax2[1].axis('off')

        plt.tight_layout()
        st.pyplot(fig2)

# ==========================================
# TAB 3: ULTRASOUND WAVE SIMULATOR
# ==========================================
with tab3:
    st.header("Ultrasound: Wave Mechanics & Reflection Coefficients")
    st.write("Ultrasound measures time delays of sound echoes across varying tissue acoustic impedances.")

    col1, col2 = st.columns([1, 2])

    with col1:
        tissue_choice = st.selectbox("Select Tissue Interface:", ["Fat to Muscle", "Muscle to Bone", "Fat to Bone"])
        frequency = st.slider("Transducer Frequency (MHz):", min_value=1.0, max_value=10.0, value=3.5, step=0.5)
        st.latex(r"d = \frac{v \cdot t}{2}")
        st.latex(r"R = \left(\frac{Z_2 - Z_1}{Z_2 + Z_1}\right)^2")

        z_dict = {"Fat": 1.38, "Muscle": 1.70, "Bone": 6.00}
        t1, t2 = tissue_choice.split(" to ")
        z1, z2 = z_dict[t1], z_dict[t2]
        r_coeff = ((z2 - z1) / (z2 + z1)) ** 2
        st.metric(label="Reflection Coefficient (R)", value=f"{r_coeff:.4f}")

    with col2:
        t_time = np.linspace(0, 10, 500)
        incident_wave = np.sin(2 * np.pi * frequency * t_time) * np.exp(-0.5 * (t_time - 2)**2)
        echo_wave = r_coeff * np.sin(2 * np.pi * frequency * t_time) * np.exp(-0.5 * (t_time - 7)**2)
        combined_signal = incident_wave + echo_wave

        fig3, ax3 = plt.subplots(figsize=(8, 3.5))
        fig3.patch.set_facecolor('#0E1117')
        ax3.set_facecolor('#0E1117')
        ax3.title.set_color('white')
        ax3.xaxis.label.set_color('white')
        ax3.yaxis.label.set_color('white')
        ax3.tick_params(colors='white')

        ax3.plot(t_time, combined_signal, color='#38BDF8', lw=1.5)
        ax3.axvline(x=2, color='#4ADE80', linestyle='--', label='Transmitter Signal')
        ax3.axvline(x=7, color='#F87171', linestyle='--', label='Boundary Echo Pulse')
        ax3.set_title("A-Scan Pulse Echo Trace")
        ax3.set_xlabel("Time Delay (microseconds)")
        ax3.set_ylabel("Amplitude")
        ax3.legend(facecolor='#0E1117', labelcolor='white')
        ax3.grid(True, alpha=0.2)

        plt.tight_layout()
        st.pyplot(fig3)

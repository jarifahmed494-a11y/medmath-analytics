# ==========================================
# TAB 1: CT SCAN SIMULATOR & LIVE X-RAY SLICES
# ==========================================
with tab1:
    st.header("CT Scan: 3D Density Map & Live X-Ray Projections")
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
        # 1. Calculate Radon transform and reconstruction based on selected angles
        theta = np.linspace(0.0, 180.0, num_angles, endpoint=False)
        sinogram = radon(base_phantom, theta=theta)
        reconstructed = iradon(sinogram, theta=theta, filter_name='ramp')

        # 2. Top Section: Matplotlib 2D Slice Figures (Reconstructed Slice & Sinogram)
        fig_xray, ax_xray = plt.subplots(1, 2, figsize=(8, 2.8))
        fig_xray.patch.set_facecolor('#0A0E17')

        for a in ax_xray:
            a.set_facecolor('#0A0E17')
            a.title.set_color('#00F0FF')
            a.tick_params(colors='white')

        # Display 2D CT Reconstructed Slice Image
        ax_xray[0].imshow(reconstructed, cmap='bone')
        ax_xray[0].set_title(f"Reconstructed Slice ({num_angles}°)")
        ax_xray[0].axis('off')

        # Display Live Radon Sinogram (X-ray sensor array data)
        ax_xray[1].imshow(sinogram, cmap='inferno', aspect='auto')
        ax_xray[1].set_title("Live Sinogram Matrix")
        ax_xray[1].axis('off')

        plt.tight_layout()
        st.pyplot(fig_xray)

        # 3. Bottom Section: 3D Density Terrain Surface (Plotly)
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

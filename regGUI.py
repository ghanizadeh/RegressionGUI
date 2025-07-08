import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, savgol_filter
import pywt

st.header("0. Preprocessing")

uploaded_file = st.file_uploader("Upload your signal dataset (CSV)", type=["csv"], key="preprocessing")

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    st.subheader("Raw Signal Preview")
    st.dataframe(df_raw.head())

    # Choose time window for plotting
    st.subheader("Plot Signals")
    sensor_cols = [col for col in df_raw.columns if col.startswith("Sensor")]
    selected_signals = st.multiselect("Select signal columns to plot", sensor_cols, default=sensor_cols[:3])
    max_rows = st.slider("Number of rows to plot (time steps)", 50, min(2000, len(df_raw)), 200)

    if selected_signals:
        fig, ax = plt.subplots(figsize=(12, 5))
        for col in selected_signals:
            ax.plot(df_raw[col].iloc[:max_rows], label=col)
        ax.set_title("Raw Signals")
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Signal Value")
        ax.legend()
        st.pyplot(fig)

    st.subheader("Apply Noise Reduction")
    apply_butter = st.checkbox("Removes high-frequency noise (Butterworth)")
    apply_ma = st.checkbox("Moving Average Filter")
    ma_window_sec = st.slider("Moving Average Window (seconds)", 1, 30, 10) if apply_ma else None
    apply_savgol = st.checkbox("Savitzky-Golay Filter")
    apply_wavelet = st.checkbox("Wavelet Denoising")

    df_processed = df_raw.copy()

    def butter_lowpass_filter(data, cutoff=1.0, fs=60.0, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)

    def wavelet_denoise(signal, wavelet='db4', level=1):
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-level])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
        coeffs[1:] = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]
        return pywt.waverec(coeffs, wavelet)[:len(signal)]

    # Apply filters
    for col in sensor_cols:
        signal = df_raw[col].values
        if apply_butter:
            signal = butter_lowpass_filter(signal)
        if apply_ma:
            window_size = int(ma_window_sec * 60)
            signal = pd.Series(signal).rolling(window=window_size, min_periods=1).mean().values
        if apply_savgol:
            try:
                signal = savgol_filter(signal, window_length=11, polyorder=2)
            except:
                pass
        if apply_wavelet:
            signal = wavelet_denoise(signal)
        df_processed[col] = signal

    if apply_butter or apply_ma or apply_savgol or apply_wavelet:
        st.subheader("Processed Signals")
        fig, ax = plt.subplots(figsize=(12, 5))
        for col in selected_signals:
            ax.plot(df_processed[col].iloc[:max_rows], label=f"{col} (Processed)")
        ax.set_title("Processed Signals")
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Signal Value")
        ax.legend()
        st.pyplot(fig)

        st.download_button("Download Processed CSV", df_processed.to_csv(index=False), file_name="processed_signals.csv")

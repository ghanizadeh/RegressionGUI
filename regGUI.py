import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.signal import savgol_filter, butter, filtfilt

# Section 0: Preprocessing
st.header("0. Preprocessing")

uploaded_signal_file = st.file_uploader("Upload signal dataset (CSV)", type=["csv"], key="signal_data")

if uploaded_signal_file:
    raw_df = pd.read_csv(uploaded_signal_file)
    st.subheader("Raw Signal Preview")
    st.dataframe(raw_df.head())

    time_col = st.selectbox("Select Timestamp Column", options=raw_df.columns, index=0)
    signal_cols = st.multiselect("Select Signal Columns", options=[col for col in raw_df.columns if col != time_col])

    if signal_cols:
        fig_raw, ax_raw = plt.subplots(figsize=(10, 4))
        for col in signal_cols:
            ax_raw.plot(raw_df[time_col], raw_df[col], label=col)
        ax_raw.set_title("Raw Signals")
        ax_raw.legend()
        st.pyplot(fig_raw)

        # Checkboxes for filtering
        apply_butter = st.checkbox("Butterworth Low-Pass Filter (removes high-frequency noise)")
        apply_moving_avg = st.checkbox("Moving Average Filter")
        apply_savgol = st.checkbox("Savitzky-Golay Filter")

        # Filtering setup
        filtered_df = raw_df.copy()

        if apply_butter:
            cutoff = st.number_input("Butterworth cutoff frequency (Hz)", min_value=0.01, value=0.1, step=0.01)
            fs = st.number_input("Sampling frequency (Hz)", min_value=1.0, value=60.0, step=1.0)
            order = 2
            b, a = butter(order, cutoff / (0.5 * fs), btype='low')
            for col in signal_cols:
                filtered_df[col] = filtfilt(b, a, filtered_df[col])

        if apply_moving_avg:
            window_sec = st.slider("Moving average window (seconds)", 1, 20, 5)
            fs = st.number_input("Sampling frequency for moving average (Hz)", min_value=1.0, value=60.0, step=1.0, key='fs_ma')
            window_size = int(window_sec * fs)
            for col in signal_cols:
                filtered_df[col] = filtered_df[col].rolling(window_size, min_periods=1, center=True).mean()

        if apply_savgol:
            window_length = st.slider("Savitzky-Golay window length (odd number)", 5, 101, 15, step=2)
            polyorder = st.slider("Polynomial order", 1, 5, 2)
            for col in signal_cols:
                filtered_df[col] = savgol_filter(filtered_df[col], window_length, polyorder)

        # Plot filtered
        st.subheader("Filtered Signals")
        fig_filtered, ax_filtered = plt.subplots(figsize=(10, 4))
        for col in signal_cols:
            ax_filtered.plot(filtered_df[time_col], filtered_df[col], label=col)
        ax_filtered.set_title("Filtered Signals")
        ax_filtered.legend()
        st.pyplot(fig_filtered)

        # Download filtered data
        st.download_button("Download Filtered Data", data=filtered_df.to_csv(index=False), file_name="filtered_signals.csv")

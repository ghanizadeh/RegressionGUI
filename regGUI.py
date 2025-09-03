import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb
import streamlit.components.v1 as components

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import PartialDependenceDisplay


#st.set_option('deprecation.showPyplotGlobalUse', False)
def plot_predicted_vs_measured_separately(y_true, y_pred, dataset_type, model_name,target):
    color = 'teal' if dataset_type == 'Train' else 'orange'

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    plt.figure(figsize=(7, 7))
    max_val = np.max(np.concatenate([y_true, y_pred]))
    min_val = 0

    plt.scatter(y_true, y_pred, c=color, edgecolors='black', marker='o' if dataset_type == 'Train' else 'v', alpha=0.7)

    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='100% agreement')
    plt.plot([min_val, max_val], [min_val, 1.2 * max_val], 'r--', linewidth=1, label='+20%')
    plt.plot([min_val, max_val], [min_val, 0.8 * max_val], 'r--', linewidth=1, label='-20%')

    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.xlabel("Measured "+target)
    plt.ylabel("Predicted "+target)
    plt.title(f"{model_name} - {dataset_type} Set")

    plt.legend(title=f"{dataset_type}:\nR²={r2:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

 
# Custom A20 index function
def a20_index(y_true, y_pred):
    ratio = y_pred / y_true
    ratio = np.where(ratio < 1, 1 / ratio, ratio)
    return np.mean(ratio <= 1.2)

def get_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    a20 = a20_index(y_true, y_pred)
    return [mae, mse, rmse, r2, a20]

def plot_pairwise_corr_with_hist(df, target_col):
    plots = []  # Collect all figures to return
    numeric_cols = df.select_dtypes(include='number').columns.drop(target_col)
    colors = plt.cm.get_cmap('Set1', len(numeric_cols)).colors

    for i, col in enumerate(numeric_cols):
        data = df[[col, target_col]].dropna()
        x = data[col]
        y = data[target_col]

        if len(data) < 2:
            continue

        try:
            corr = np.corrcoef(x, y)[0, 1]
            r2 = corr ** 2
            slope, intercept = np.polyfit(x, y, 1)
            line_x = np.linspace(x.min(), x.max(), 100)
            line_y = slope * line_x + intercept
        except np.linalg.LinAlgError:
            continue

        fig = plt.figure(figsize=(7, 7))
        grid = plt.GridSpec(4, 4, hspace=0.05, wspace=0.05)
        ax_main = fig.add_subplot(grid[1:4, 0:3])
        ax_xhist = fig.add_subplot(grid[0, 0:3], sharex=ax_main)
        ax_yhist = fig.add_subplot(grid[1:4, 3], sharey=ax_main)

        ax_main.scatter(x, y, color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.5)
        ax_main.plot(line_x, line_y, color='red', linestyle='--', linewidth=2, label=f'R² = {r2:.2f}')
        ax_main.legend(loc='upper center')
        ax_main.set_xlabel(col, fontsize=11)
        ax_main.set_ylabel(target_col, fontsize=11)

        counts_x, bins_x, _ = ax_xhist.hist(x, bins=15, color='green', edgecolor='black')
        for j in range(len(bins_x) - 1):
            if counts_x[j] > 0:
                ax_xhist.text((bins_x[j] + bins_x[j+1]) / 2, counts_x[j], str(int(counts_x[j])), ha='center', va='bottom', fontsize=10)
        ax_xhist.axis('off')

        counts_y, bins_y, _ = ax_yhist.hist(y, bins=15, orientation='horizontal', color='green', edgecolor='black')
        for j in range(len(bins_y) - 1):
            if counts_y[j] > 0:
                ax_yhist.text(counts_y[j] + max(counts_y)*0.02, (bins_y[j] + bins_y[j+1]) / 2, str(int(counts_y[j])), va='center', ha='left', fontsize=10)
        ax_yhist.axis('off')

        plots.append(fig)  # Store figure

    return plots

# Streamlit App
st.title("MLCan Simple AI")

# Upload CSV
st.sidebar.header("Upload Dataset:")

uploaded_file = st.sidebar.file_uploader("Upload your file (csv)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.header("1. Select Features and Target")
    with st.expander("Column Selection"):
        cols = df.columns.tolist()
        selected_features = st.multiselect("Select Feature Columns", options=cols)
        selected_target = st.selectbox("Select Target Column", options=[col for col in cols if col not in selected_features])

    if selected_features and selected_target:
        X = df[selected_features]
        y = df[selected_target]

        st.header("2. Exploratory Data Analysis")
        with st.expander("Show/hide EDM", expanded=True):
            st.subheader("2. Summary Statistics with Skew & Kurtosis")
            summary = df[selected_features + [selected_target]].describe().T
            summary['skew'] = df[selected_features + [selected_target]].skew()
            summary['kurtosis'] = df[selected_features + [selected_target]].kurtosis()
            st.dataframe(summary.round(3))
    
            st.subheader("Scatter Plots + Histograms")
        
            plots = plot_pairwise_corr_with_hist(df[selected_features + [selected_target]], selected_target)
            for fig in plots:
                st.pyplot(fig)
    
            st.subheader("Correlation Heatmap")
            #fig, ax = plt.subplots()
            #sns.heatmap(df[selected_features + [selected_target]].corr(), annot=True, cmap='coolwarm', ax=ax)
            #st.pyplot(fig)
    
            # Compute correlation matrix
            corr = df[selected_features + [selected_target]].corr()
            # Create a mask for the upper triangle
            mask = np.triu(np.ones_like(corr, dtype=bool))
            
            # Bigger figure
            fig, ax = plt.subplots(figsize=(10, 8))  
            
            # Heatmap
            sns.heatmap(
                corr,
                mask=mask,
                annot=True,
                fmt=".2f",           # Limit decimals
                cmap="coolwarm",
                cbar=True,
                ax=ax,
                square=True,
                linewidths=0.5,
                annot_kws={"size": 9}  # smaller text inside cells
            )
            
            # Rotate x-axis labels and move them to top
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="left")
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.set_label_position('top')
            
            st.pyplot(fig)




        
        st.header("3. Model Training and Evaluation")
        model_choice = st.radio("Select Model", ["Random Forest", "XGBoost"])
        eval_method = st.radio("Evaluation Method", ["Train-Test Split", "K-Fold Cross-Validation"])

        if model_choice == "Random Forest":
            base_model = RandomForestRegressor(random_state=42)
        else:
            base_model = xgb.XGBRegressor(random_state=42)

        if eval_method == "Train-Test Split":
            test_size = st.slider("Test Size (%)", 10, 50, 25) / 100
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            model = base_model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_metrics = get_metrics(y_train, y_train_pred)
            test_metrics = get_metrics(y_test, y_test_pred)

            result_df = pd.DataFrame([train_metrics, test_metrics],
                                     columns=["MAE", "MSE", "RMSE", "R²", "A20 Index"],
                                     index=["Train", "Test"])
            st.subheader("Model Performance")
            st.dataframe(result_df.round(4))
                    # Add Predicted vs Measured Plot
            st.subheader("Predicted vs. Measured")

            # Call the function you provided
            st.markdown("##### 🔵 Training Set")
            plot_predicted_vs_measured_separately(y_train, y_train_pred, "Train", model_choice, selected_target)

            st.markdown("##### 🟠 Testing Set")
            plot_predicted_vs_measured_separately(y_test, y_test_pred, "Test", model_choice, selected_target)
        else:  # K-Fold
            k = st.slider("Number of Folds (K)", 2, 10, 5)
            use_holdout = st.checkbox("Include Hold-out Test Set?")
            if use_holdout:
                holdout_size = st.slider("Hold-out Test Size (%)", 10, 30, 20) / 100
                X_main, X_holdout, y_main, y_holdout = train_test_split(X, y, test_size=holdout_size, random_state=42)
            else:
                X_main, y_main = X, y

            kf = KFold(n_splits=k, shuffle=True, random_state=42)
            train_scores, val_scores = [], []

            # For plotting
            all_train_true, all_train_pred = [], []
            all_val_true, all_val_pred = [], []

            for train_idx, val_idx in kf.split(X_main):
                X_train, X_val = X_main.iloc[train_idx], X_main.iloc[val_idx]
                y_train, y_val = y_main.iloc[train_idx], y_main.iloc[val_idx]

                model = base_model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)

                # Save metrics
                train_scores.append(get_metrics(y_train, y_train_pred))
                val_scores.append(get_metrics(y_val, y_val_pred))

                # For plot
                all_train_true.extend(y_train)
                all_train_pred.extend(y_train_pred)
                all_val_true.extend(y_val)
                all_val_pred.extend(y_val_pred)

            # Average metrics
            avg_train = np.mean(train_scores, axis=0)
            avg_val = np.mean(val_scores, axis=0)

            results = {
                "Train (CV Avg)": avg_train,
                "Validation (CV Avg)": avg_val
            }

            # Hold-out evaluation
            if use_holdout:
                model = base_model.fit(X_main, y_main)
                y_holdout_pred = model.predict(X_holdout)
                results["Test (Hold-out)"] = get_metrics(y_holdout, y_holdout_pred)

            # Display performance
            results_df = pd.DataFrame(results, index=["MAE", "MSE", "RMSE", "R²", "A20 Index"]).T
            st.subheader("Model Performance")
            st.dataframe(results_df.round(4))

            # Display Predicted vs Measured
            st.subheader("Predicted vs. Measured")

            st.markdown("##### 🔵 Training Set (CV Aggregated)")
            plot_predicted_vs_measured_separately(np.array(all_train_true), np.array(all_train_pred), "Train", model_choice, selected_target)

            st.markdown("##### 🟠 Validation Set (CV Aggregated)")
            plot_predicted_vs_measured_separately(np.array(all_val_true), np.array(all_val_pred), "Validation", model_choice, selected_target)

            if use_holdout:
                st.markdown("##### 🧪 Hold-out Test Set")
                plot_predicted_vs_measured_separately(y_holdout, y_holdout_pred, "Test (Holdout)", model_choice, selected_target)

        # SHAP + PDP Section
        st.header("4. Model Interpretation")
        
        show_shap = st.checkbox("4.1. Show SHAP Plots (Global & Local)")
        show_pdp = st.checkbox("4.2. Show Partial Dependence Plot (PDP)")
        if show_shap:
            with st.expander("SHAP Waterfall + Beeswarm Plots"):
                explainer = shap.Explainer(model)
                try:
                    shap_values = explainer(X_test if eval_method == "Train-Test Split" else X_holdout if use_holdout else X_main)
                    sample_idx = 0
                    st.write("Showing SHAP values for instance index:", sample_idx)
                    fig = plt.figure()
                    shap.plots.waterfall(shap_values[sample_idx], max_display=10, show=False)
                    st.pyplot(fig)
                    fig2 = plt.figure()
                    shap.plots.beeswarm(shap_values, show=False)
                    st.pyplot(fig2)
                except Exception as e:
                    st.warning(f"SHAP plot failed: {e}")

        if show_pdp:
            with st.expander("📉 Partial Dependence Plot (PDP)"):
                pdp_feature = st.selectbox("Select feature for PDP", selected_features)
                fig, ax = plt.subplots()
                PartialDependenceDisplay.from_estimator(model, X, [pdp_feature], ax=ax)
                st.pyplot(fig)

    else:
        st.warning("Please select both feature(s) and a target column.")

    # 5️⃣ Final Model Testing on New Dataset
    st.header("5. Test Final Model on New Dataset")

    if 'model' in locals():
        new_data_file = st.file_uploader("Upload a new dataset for prediction (CSV)", type=["csv"], key="new_dataset")

        if new_data_file:
            new_df = pd.read_csv(new_data_file)
            missing_cols = [col for col in selected_features + [selected_target] if col not in new_df.columns]

            if missing_cols:
                st.warning(f"Missing required columns in uploaded file: {missing_cols}")
            else:
                new_X = new_df[selected_features]
                new_y_true = new_df[selected_target]
                new_y_pred = model.predict(new_X)

                # Compute metrics
                new_metrics = get_metrics(new_y_true, new_y_pred)
                metrics_df = pd.DataFrame([new_metrics], columns=["MAE", "MSE", "RMSE", "R²", "A20 Index"], index=["New Dataset"])
                st.subheader("Performance on New Dataset")
                st.dataframe(metrics_df.round(4))

                # Predicted vs Measured Plot
                st.subheader("Predicted vs. Measured (New Dataset)")
                plot_predicted_vs_measured_separately(new_y_true, new_y_pred, "New", model_choice, selected_target)

    else:
        st.info("Please train a model in Section 3 before testing it on new data.")





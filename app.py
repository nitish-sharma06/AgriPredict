import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    roc_auc_score,
)

# ============================================================
# AgriPredict - Streamlit Application
# Based on the project's existing notebook and dataset
# ============================================================

st.set_page_config(
    page_title="AgriPredict | Agricultural Analytics",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILE = "seasonal_agriculture_data.csv"

FEATURE_COLS = [
    "Season",
    "State",
    "Crop",
    "Farm_Area_Hectares",
    "Rainfall_mm",
    "Avg_Temperature_C",
    "Humidity_pct",
    "Sunlight_Hours_Day",
    "Soil_pH",
    "Soil_Moisture_pct",
    "Nitrogen_kg_ha",
    "Phosphorus_kg_ha",
    "Potassium_kg_ha",
    "Irrigation_Method",
    "Fertilizer_kg_ha",
    "Pesticide_Litre_ha",
    "Seed_Quality_Score",
    "Disease_Pest_Risk_pct",
]

CATEGORICAL_COLS = [
    "Season",
    "State",
    "Crop",
    "Irrigation_Method",
]


# -----------------------------
# Helper functions
# -----------------------------
def money_inr(value):
    if pd.isna(value):
        return "₹0"
    value = float(value)
    sign = "-" if value < 0 else ""
    value = abs(value)

    if value >= 1e7:
        return f"{sign}₹{value/1e7:.2f} Cr"
    if value >= 1e5:
        return f"{sign}₹{value/1e5:.2f} L"
    if value >= 1e3:
        return f"{sign}₹{value/1e3:.1f}K"
    return f"{sign}₹{value:,.0f}"


def load_and_prepare_data():
    if not os.path.exists(DATA_FILE):
        st.error(
            f"Could not find `{DATA_FILE}`. "
            "Make sure app.py and the CSV file are in the same GitHub folder."
        )
        st.stop()

    df = pd.read_csv(DATA_FILE)

    required = [
        "Farm_Area_Hectares",
        "Production_Tonnes",
        "Yield_Tonnes_Ha",
        "Rainfall_mm",
        "Soil_Moisture_pct",
        "Profit_INR",
        "Revenue_INR",
        "Total_Cost_INR",
        "Nitrogen_kg_ha",
        "Phosphorus_kg_ha",
        "Potassium_kg_ha",
    ]

    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        st.error(f"Dataset is missing required columns: {missing_cols}")
        st.stop()

    # Same domain-grounded imputation logic used in the notebook.
    missing_yield = df["Yield_Tonnes_Ha"].isna()
    df.loc[missing_yield, "Yield_Tonnes_Ha"] = (
        df.loc[missing_yield, "Production_Tonnes"]
        / df.loc[missing_yield, "Farm_Area_Hectares"]
    ).round(2)

    rainfall_group = df.groupby(["Season", "District"])["Rainfall_mm"].transform("median")
    df["Rainfall_mm"] = df["Rainfall_mm"].fillna(rainfall_group)
    df["Rainfall_mm"] = df["Rainfall_mm"].fillna(
        df.groupby("Season")["Rainfall_mm"].transform("median")
    )
    df["Rainfall_mm"] = df["Rainfall_mm"].fillna(df["Rainfall_mm"].median())

    moisture_group = df.groupby(
        ["Season", "Irrigation_Method"]
    )["Soil_Moisture_pct"].transform("median")
    df["Soil_Moisture_pct"] = df["Soil_Moisture_pct"].fillna(moisture_group)
    df["Soil_Moisture_pct"] = df["Soil_Moisture_pct"].fillna(
        df["Soil_Moisture_pct"].median()
    )

    # Notebook-derived features.
    df["Profit_Margin_pct"] = np.where(
        df["Revenue_INR"] > 0,
        (df["Profit_INR"] / df["Revenue_INR"]) * 100,
        -100.0,
    )
    df["Cost_Per_Hectare"] = df["Total_Cost_INR"] / df["Farm_Area_Hectares"]
    df["Revenue_Per_Hectare"] = df["Revenue_INR"] / df["Farm_Area_Hectares"]
    df["Profit_Per_Hectare"] = df["Profit_INR"] / df["Farm_Area_Hectares"]
    df["Financial_Status"] = np.where(
        df["Profit_INR"] > 0, "Profitable", "Loss-Making"
    )
    df["Is_Profitable"] = (df["Profit_INR"] > 0).astype(int)
    df["NPK_Total"] = (
        df["Nitrogen_kg_ha"]
        + df["Phosphorus_kg_ha"]
        + df["Potassium_kg_ha"]
    )

    return df


@st.cache_data
def get_data():
    return load_and_prepare_data()


@st.cache_resource
def train_models(df):
    # Match the notebook's feature set and one-hot encoding.
    X = df[FEATURE_COLS].copy()
    X = pd.get_dummies(
        X,
        columns=CATEGORICAL_COLS,
        drop_first=True,
    )

    # Fill any numerical missing values defensively.
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)

    # -----------------------------
    # Yield model
    # -----------------------------
    reg_mask = df["Crop"] != "Sugarcane"
    X_reg = X.loc[reg_mask]
    y_reg = df.loc[reg_mask, "Yield_Tonnes_Ha"]

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg,
        y_reg,
        test_size=0.20,
        random_state=42,
    )

    rf_reg = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    rf_reg.fit(X_train_r, y_train_r)

    pred_r = rf_reg.predict(X_test_r)

    yield_metrics = {
        "r2": r2_score(y_test_r, pred_r),
        "rmse": np.sqrt(mean_squared_error(y_test_r, pred_r)),
        "mae": mean_absolute_error(y_test_r, pred_r),
    }

    # -----------------------------
    # Profitability model
    # -----------------------------
    y_clf = df["Is_Profitable"]

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X,
        y_clf,
        test_size=0.20,
        random_state=42,
        stratify=y_clf,
    )

    rf_clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    rf_clf.fit(X_train_c, y_train_c)

    pred_c = rf_clf.predict(X_test_c)
    prob_c = rf_clf.predict_proba(X_test_c)[:, 1]

    clf_metrics = {
        "accuracy": accuracy_score(y_test_c, pred_c),
        "roc_auc": roc_auc_score(y_test_c, prob_c),
    }

    return (
        rf_reg,
        rf_clf,
        list(X.columns),
        yield_metrics,
        clf_metrics,
    )


def encode_user_input(input_df, model_columns):
    X_new = input_df[FEATURE_COLS].copy()

    X_new = pd.get_dummies(
        X_new,
        columns=CATEGORICAL_COLS,
        drop_first=True,
    )

    # Make the input feature columns identical to training columns.
    X_new = X_new.reindex(columns=model_columns, fill_value=0)

    X_new = X_new.replace([np.inf, -np.inf], np.nan)
    X_new = X_new.fillna(0)

    return X_new


# -----------------------------
# Load data/models
# -----------------------------
df = get_data()

rf_reg, rf_clf, model_columns, yield_metrics, clf_metrics = train_models(df)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🌾 AgriPredict")
st.sidebar.caption("Agricultural Prediction & Analytics")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Dashboard",
        "🌱 Crop Recommendation",
        "📈 Yield Prediction",
        "💰 Profitability Prediction",
        "📊 Data Explorer",
        "🤖 Model Performance",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Educational/research application. Predictions should be "
    "validated with local agricultural and expert knowledge."
)

# ============================================================
# Dashboard
# ============================================================
if page == "🏠 Dashboard":
    st.title("🌾 AgriPredict")
    st.subheader("Data-driven agricultural prediction and analytics")

    st.write(
        "Explore seasonal agricultural patterns, estimate crop yield, "
        "compare crop options, and classify potential farm profitability."
    )

    total_farms = len(df)
    total_states = df["State"].nunique()
    total_crops = df["Crop"].nunique()
    avg_yield = df["Yield_Tonnes_Ha"].mean()
    total_profit = df["Profit_INR"].sum()
    profitable_pct = df["Is_Profitable"].mean() * 100

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Farm Records", f"{total_farms:,}")
    c2.metric("States", total_states)
    c3.metric("Crop Types", total_crops)
    c4.metric("Avg Yield", f"{avg_yield:.2f} t/ha")
    c5.metric("Profitable Farms", f"{profitable_pct:.1f}%")

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.subheader("Average Yield by Season")
        season_yield = (
            df.groupby("Season", as_index=False)["Yield_Tonnes_Ha"]
            .mean()
            .sort_values("Yield_Tonnes_Ha", ascending=False)
        )
        st.bar_chart(
            season_yield.set_index("Season")["Yield_Tonnes_Ha"]
        )

    with right:
        st.subheader("Average Profit by Season")
        season_profit = (
            df.groupby("Season", as_index=False)["Profit_INR"]
            .mean()
        )
        st.bar_chart(
            season_profit.set_index("Season")["Profit_INR"]
        )

    st.subheader("Top Crops by Average Yield")
    crop_yield = (
        df.groupby("Crop")["Yield_Tonnes_Ha"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    st.bar_chart(crop_yield)

    st.subheader("Irrigation Method Performance")
    irrigation = (
        df.groupby("Irrigation_Method")
        .agg(
            Avg_Yield=("Yield_Tonnes_Ha", "mean"),
            Avg_Water_Efficiency=(
                "Water_Efficiency_t_per_1000m3",
                "mean",
            ),
            Avg_Profit=("Profit_INR", "mean"),
        )
        .round(2)
    )
    st.dataframe(irrigation, use_container_width=True)


# ============================================================
# Common input UI
# ============================================================
elif page in [
    "🌱 Crop Recommendation",
    "📈 Yield Prediction",
    "💰 Profitability Prediction",
]:
    st.title(page)

    st.write(
        "Enter farm and environmental conditions. The application uses "
        "the Random Forest models developed from the AgriPredict dataset."
    )

    states = sorted(df["State"].dropna().unique().tolist())
    seasons = sorted(df["Season"].dropna().unique().tolist())
    crops = sorted(df["Crop"].dropna().unique().tolist())
    irrigation_methods = sorted(
        df["Irrigation_Method"].dropna().unique().tolist()
    )

    # Reasonable dataset-based defaults/ranges.
    defaults = {
        "area": float(df["Farm_Area_Hectares"].median()),
        "rainfall": float(df["Rainfall_mm"].median()),
        "temperature": float(df["Avg_Temperature_C"].median()),
        "humidity": float(df["Humidity_pct"].median()),
        "sunlight": float(df["Sunlight_Hours_Day"].median()),
        "soil_ph": float(df["Soil_pH"].median()),
        "moisture": float(df["Soil_Moisture_pct"].median()),
        "nitrogen": float(df["Nitrogen_kg_ha"].median()),
        "phosphorus": float(df["Phosphorus_kg_ha"].median()),
        "potassium": float(df["Potassium_kg_ha"].median()),
        "fertilizer": float(df["Fertilizer_kg_ha"].median()),
        "pesticide": float(df["Pesticide_Litre_ha"].median()),
        "seed_quality": float(df["Seed_Quality_Score"].median()),
        "risk": float(df["Disease_Pest_Risk_pct"].median()),
    }

    st.markdown("### 1. Farm & Crop Information")

    a, b, c, d = st.columns(4)

    with a:
        season = st.selectbox("Season", seasons)
    with b:
        state = st.selectbox("State", states)
    with c:
        selected_crop = st.selectbox("Crop", crops)
    with d:
        irrigation = st.selectbox(
            "Irrigation Method",
            irrigation_methods,
        )

    area = st.number_input(
        "Farm Area (hectares)",
        min_value=0.01,
        value=max(0.01, round(defaults["area"], 2)),
        step=0.1,
    )

    st.markdown("### 2. Environmental Conditions")

    a, b, c, d = st.columns(4)

    with a:
        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            value=round(defaults["rainfall"], 1),
        )
    with b:
        temperature = st.number_input(
            "Average Temperature (°C)",
            value=round(defaults["temperature"], 1),
        )
    with c:
        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=round(defaults["humidity"], 1),
        )
    with d:
        sunlight = st.number_input(
            "Sunlight (hours/day)",
            min_value=0.0,
            value=round(defaults["sunlight"], 1),
        )

    st.markdown("### 3. Soil & Nutrients")

    a, b, c, d = st.columns(4)

    with a:
        soil_ph = st.number_input(
            "Soil pH",
            min_value=0.0,
            max_value=14.0,
            value=round(defaults["soil_ph"], 2),
        )
    with b:
        moisture = st.number_input(
            "Soil Moisture (%)",
            min_value=0.0,
            max_value=100.0,
            value=round(defaults["moisture"], 1),
        )
    with c:
        nitrogen = st.number_input(
            "Nitrogen (kg/ha)",
            min_value=0.0,
            value=round(defaults["nitrogen"], 1),
        )
    with d:
        phosphorus = st.number_input(
            "Phosphorus (kg/ha)",
            min_value=0.0,
            value=round(defaults["phosphorus"], 1),
        )

    a, b, c, d = st.columns(4)

    with a:
        potassium = st.number_input(
            "Potassium (kg/ha)",
            min_value=0.0,
            value=round(defaults["potassium"], 1),
        )
    with b:
        fertilizer = st.number_input(
            "Fertilizer (kg/ha)",
            min_value=0.0,
            value=round(defaults["fertilizer"], 1),
        )
    with c:
        pesticide = st.number_input(
            "Pesticide (litre/ha)",
            min_value=0.0,
            value=round(defaults["pesticide"], 2),
        )
    with d:
        seed_quality = st.number_input(
            "Seed Quality Score",
            min_value=0.0,
            max_value=1.0,
            value=round(defaults["seed_quality"], 2),
        )

    disease_risk = st.slider(
        "Disease / Pest Risk (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(round(defaults["risk"], 1)),
    )

    input_row = pd.DataFrame(
        [
            {
                "Season": season,
                "State": state,
                "Crop": selected_crop,
                "Farm_Area_Hectares": area,
                "Rainfall_mm": rainfall,
                "Avg_Temperature_C": temperature,
                "Humidity_pct": humidity,
                "Sunlight_Hours_Day": sunlight,
                "Soil_pH": soil_ph,
                "Soil_Moisture_pct": moisture,
                "Nitrogen_kg_ha": nitrogen,
                "Phosphorus_kg_ha": phosphorus,
                "Potassium_kg_ha": potassium,
                "Irrigation_Method": irrigation,
                "Fertilizer_kg_ha": fertilizer,
                "Pesticide_Litre_ha": pesticide,
                "Seed_Quality_Score": seed_quality,
                "Disease_Pest_Risk_pct": disease_risk,
            }
        ]
    )

    encoded_input = encode_user_input(input_row, model_columns)

    # ========================================================
    # Crop Recommendation
    # ========================================================
    if page == "🌱 Crop Recommendation":
        st.markdown("---")
        st.subheader("🌱 Crop Recommendation")

        st.write(
            "The app tests each available crop under the selected "
            "environmental and farm conditions and ranks crops by "
            "predicted yield. Sugarcane is excluded from the yield "
            "model to match the project's notebook methodology."
        )

        if st.button("🔍 Recommend Crops", type="primary"):
            recommendations = []

            for crop in crops:
                if crop == "Sugarcane":
                    continue

                candidate = input_row.copy()
                candidate["Crop"] = crop
                encoded_candidate = encode_user_input(
                    candidate, model_columns
                )

                predicted_yield = float(
                    rf_reg.predict(encoded_candidate)[0]
                )

                recommendations.append(
                    {
                        "Crop": crop,
                        "Predicted Yield (t/ha)": predicted_yield,
                    }
                )

            result = (
                pd.DataFrame(recommendations)
                .sort_values(
                    "Predicted Yield (t/ha)",
                    ascending=False,
                )
                .reset_index(drop=True)
            )

            result.index = result.index + 1

            st.success(
                f"Highest predicted yield in this scenario: "
                f"**{result.iloc[0]['Crop']}** "
                f"({result.iloc[0]['Predicted Yield (t/ha)']:.2f} t/ha)"
            )

            st.dataframe(
                result.style.format(
                    {"Predicted Yield (t/ha)": "{:.2f}"}
                ),
                use_container_width=True,
            )

            st.bar_chart(
                result.set_index("Crop")[
                    "Predicted Yield (t/ha)"
                ]
            )

            st.caption(
                "This is a model-based ranking, not a guarantee of actual "
                "farm performance. Consider local soil, market, water, "
                "weather and expert agricultural advice."
            )

    # ========================================================
    # Yield Prediction
    # ========================================================
    elif page == "📈 Yield Prediction":
        st.markdown("---")
        st.subheader("📈 Crop Yield Prediction")

        if selected_crop == "Sugarcane":
            st.warning(
                "The notebook's yield regression model excludes Sugarcane. "
                "Please select another crop for a model prediction."
            )

        if st.button("📈 Predict Yield", type="primary"):
            if selected_crop == "Sugarcane":
                st.stop()

            predicted_yield = float(
                rf_reg.predict(encoded_input)[0]
            )

            estimated_production = predicted_yield * area

            c1, c2 = st.columns(2)
            c1.metric(
                "Predicted Yield",
                f"{predicted_yield:.2f} tonnes/ha",
            )
            c2.metric(
                "Estimated Production",
                f"{estimated_production:.2f} tonnes",
            )

            st.info(
                f"For **{selected_crop}** on **{area:.2f} hectares**, "
                f"the model predicts approximately "
                f"**{predicted_yield:.2f} tonnes/ha**."
            )

            st.caption(
                "Prediction is based on the project's Random Forest "
                "yield model and should be treated as an analytical estimate."
            )

    # ========================================================
    # Profitability Prediction
    # ========================================================
    else:
        st.markdown("---")
        st.subheader("💰 Farm Profitability Prediction")

        if st.button("💰 Predict Profitability", type="primary"):
            prediction = int(rf_clf.predict(encoded_input)[0])
            probability = float(
                rf_clf.predict_proba(encoded_input)[0, 1]
            )

            if prediction == 1:
                status = "Profitable"
                st.success("### 🟢 Predicted Status: Profitable")
            else:
                status = "Loss-Making"
                st.error("### 🔴 Predicted Status: Loss-Making")

            c1, c2 = st.columns(2)
            c1.metric(
                "Probability of Profitability",
                f"{probability * 100:.1f}%",
            )
            c2.metric(
                "Model Classification",
                status,
            )

            st.progress(min(max(probability, 0.0), 1.0))

            st.caption(
                "This classifier predicts the probability of positive "
                "net profit from the project's historical dataset. "
                "It does not predict an exact rupee profit."
            )


# ============================================================
# Data Explorer
# ============================================================
elif page == "📊 Data Explorer":
    st.title("📊 Agricultural Data Explorer")

    st.write(
        "Explore the 4,000 farm observations used by the AgriPredict project."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        selected_seasons = st.multiselect(
            "Filter by Season",
            sorted(df["Season"].unique()),
            default=sorted(df["Season"].unique()),
        )

    with c2:
        selected_states = st.multiselect(
            "Filter by State",
            sorted(df["State"].unique()),
            default=sorted(df["State"].unique()),
        )

    with c3:
        selected_crops = st.multiselect(
            "Filter by Crop",
            sorted(df["Crop"].unique()),
            default=sorted(df["Crop"].unique()),
        )

    filtered = df[
        df["Season"].isin(selected_seasons)
        & df["State"].isin(selected_states)
        & df["Crop"].isin(selected_crops)
    ]

    st.metric("Filtered Records", f"{len(filtered):,}")

    display_cols = [
        "Farm_ID",
        "State",
        "District",
        "Crop",
        "Season",
        "Farm_Area_Hectares",
        "Rainfall_mm",
        "Avg_Temperature_C",
        "Yield_Tonnes_Ha",
        "Profit_INR",
        "Irrigation_Method",
        "Water_Efficiency_t_per_1000m3",
    ]

    st.dataframe(
        filtered[display_cols],
        use_container_width=True,
        height=500,
    )

    st.subheader("Summary Statistics")

    summary = (
        filtered[
            [
                "Farm_Area_Hectares",
                "Rainfall_mm",
                "Avg_Temperature_C",
                "Yield_Tonnes_Ha",
                "Profit_INR",
                "Water_Efficiency_t_per_1000m3",
            ]
        ]
        .describe()
        .T
        .round(2)
    )

    st.dataframe(summary, use_container_width=True)


# ============================================================
# Model Performance
# ============================================================
else:
    st.title("🤖 Model Performance")

    st.write(
        "Performance below is calculated when the Streamlit app trains "
        "the models using the same core methodology as the project notebook."
    )

    st.subheader("🌾 Yield Regression Model")

    c1, c2, c3 = st.columns(3)

    c1.metric("R² Score", f"{yield_metrics['r2']:.3f}")
    c2.metric("RMSE", f"{yield_metrics['rmse']:.3f} t/ha")
    c3.metric("MAE", f"{yield_metrics['mae']:.3f} t/ha")

    st.markdown("---")

    st.subheader("💰 Profitability Classification Model")

    c1, c2 = st.columns(2)

    c1.metric(
        "Accuracy",
        f"{clf_metrics['accuracy'] * 100:.2f}%",
    )
    c2.metric(
        "ROC-AUC",
        f"{clf_metrics['roc_auc']:.3f}",
    )

    st.markdown("---")

    st.subheader("Model Configuration")

    model_info = pd.DataFrame(
        {
            "Model": [
                "Yield Prediction",
                "Profitability Prediction",
            ],
            "Algorithm": [
                "Random Forest Regressor",
                "Random Forest Classifier",
            ],
            "Estimators": [150, 150],
            "Max Depth": [12, 10],
            "Random State": [42, 42],
        }
    )

    st.dataframe(model_info, use_container_width=True)

    st.info(
        "The original notebook reports approximately R² = 0.896 for the "
        "yield regressor and 83.50% accuracy / 0.913 ROC-AUC for the "
        "profitability classifier on its recorded test split. "
        "The values shown above are recomputed when the app runs."
    )


# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption(
    "AgriPredict • Agricultural Prediction & Data Analytics • "
    "Educational / Research Project"
)

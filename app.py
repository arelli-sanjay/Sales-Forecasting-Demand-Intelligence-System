import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb


st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    layout="wide"
)

st.markdown("""
<style>

/* SIDEBAR BACKGROUND */

[data-testid="stSidebar"]{
    border-right: 1px solid rgba(255,255,255,0.08);        
}

/* APP LOGO / TITLE */

.sidebar-title{
    color:black;
    font-size:24px;
    font-weight:700;
    text-align:left;
    padding-top:10px;
    padding-bottom:5px;
}

.sidebar-subtitle{
    color:#B8C5D6;
    font-size:13px;
    margin-top:-8px;
    margin-bottom:18px;
}

/* RADIO LABEL */

div[data-testid="stRadio"] > label{
    color:#7F95B2 !important;
    font-size:12px !important;
    font-weight:700 !important;
    letter-spacing:1px;
    text-transform:uppercase;
    margin-top:10px;
    margin-bottom:8px;
}

/* RADIO BUTTONS */

div[role="radiogroup"] label{
    background: transparent;
    color:white !important;
    border-radius:12px;
    padding:14px 18px;
    margin-bottom:10px;
    transition:0.3s;
    font-size:17px;
    font-weight:500;
    border:1px solid transparent;
}

/* Hover */

div[role="radiogroup"] label:hover{
    background:#14346E;
    cursor:pointer;
}

/* Selected */

div[role="radiogroup"] label:has(input:checked){
    background:linear-gradient(135deg,#2C63F3,#1847C6);
    border-radius:12px;
    color:white !important;
    box-shadow:0 8px 20px rgba(0,0,0,.25);
}

/* Hide default radio circles */

div[role="radiogroup"] input[type="radio"]{
    display:none;
}

/* DIVIDER */

hr{
    border:0;
    border-top:1px solid rgba(255,255,255,.08);
}

/* METRICS */

div[data-testid="metric-container"]{
    border-radius:14px;
    padding:15px;
    background:white;
    box-shadow:0 5px 18px rgba(0,0,0,.08);
}

/* BUTTONS */

.stButton>button{
    border-radius:10px;
    border:none;
    background:#2563EB;
    color:white;
    font-weight:600;
}

.stButton>button:hover{
    background:#1847C6;
    color:white;
}

</style>
""", unsafe_allow_html=True)


col1, col2 = st.columns([1, 8])
with col1:
     st.markdown("<div style='margin-top:25px;'>", unsafe_allow_html=True)
     st.image("images/logo.png", width=80)
     st.markdown("</div>", unsafe_allow_html=True)
with col2:
    st.title("Sales Forecasting & Demand Intelligence System")

model = xgb.XGBRegressor()
model.load_model("models/xgb_model.json")

# Load datasets
df = pd.read_csv(
    "datasets/train_processed.csv",
    parse_dates=["Order Date"]
)

monthly_sales = pd.read_csv("datasets/monthly_sales.csv")
weekly_sales = pd.read_csv(
    "datasets/weekly_sales.csv",
    parse_dates=["Order Date"]
)

anomalies = pd.read_csv(
    "datasets/anomalies.csv",
    parse_dates=["Order Date"]
)

forecast_results = pd.read_csv("datasets/forecast_results.csv")

subcategory_features = pd.read_csv("datasets/subcategory_features.csv")

season_mapping = {
    "Winter": 0,
    "Spring": 1,
    "Summer": 2,
    "Autumn": 3
}

def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"


# Sidebar Navigation

st.sidebar.title(" Main Menu")

page = st.sidebar.radio(
    "Navigation",
    (
        "Sales Overview Dashboard",
        "Forecast Explorer",
        "Anomaly Report",
        "Product Demand Segments"
    ),
    label_visibility="collapsed"
)

# Page Routing

if page == "Sales Overview Dashboard":

    total_sales = df["Sales"].sum()

    st.metric(
        label="Total Sales",
        value=f"${total_sales:,.2f}"
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        region = st.selectbox(
            "Select Region",
            ["All"] + list(df["Region"].unique())
        )

    with col2:
        category = st.selectbox(
            "Select Category",
            ["All"] + list(df["Category"].unique())
        )

    filtered_df = df.copy()

    if region != "All":
        filtered_df = filtered_df[
            filtered_df["Region"] == region
        ]

    if category != "All":
        filtered_df = filtered_df[
            filtered_df["Category"] == category
        ]

    st.subheader("Total Sales by Year")

    yearly_sales = (
        filtered_df
        .groupby("Year")["Sales"]
        .sum()
    )

    st.bar_chart(yearly_sales)

    st.subheader("Monthly Sales Trend")

    monthly = (
        filtered_df
        .set_index("Order Date")
        .resample("ME")["Sales"]
        .sum()
    )

    st.line_chart(monthly)

elif page == "Forecast Explorer":

    prediction_type = st.selectbox(
        "Forecast By",
        ["Category", "Region"]
    )

    if prediction_type == "Category":
        option = st.selectbox(
            "Select Category",
            [
                "Furniture",
                "Technology",
                "Office Supplies"
            ]
        )
    else:
        option = st.selectbox(
            "Select Region",
            [
                "West",
                "East"
            ]
        )

    months = st.slider(
        "Forecast Horizon (Months)",
        min_value=1,
        max_value=3,
        value=3
    )

    if st.button("Generate Forecast"):
        result = forecast_results[
            forecast_results["Segment"] == option
        ]

        predictions = []

        for i in range(1, months + 1):
            predictions.append(
                result[f"Month{i}"].values[0]
            )

        forecast_df = pd.DataFrame({
            "Forecast Month": [
                f"Month {i}"
                for i in range(1, months + 1)
            ],
            "Predicted Sales": predictions
        })

        st.subheader("Forecast Results")

        st.dataframe(
            forecast_df,
            width="stretch"
        )

        st.subheader("Forecast Trend")

        chart = forecast_df.set_index(
            "Forecast Month"
        )

        st.line_chart(chart)

        st.subheader("Model Performance")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "MAE",
                "8619.70"
            )

        with col2:
            st.metric(
                "RMSE",
                "11041.84"
            )

elif page == "Anomaly Report":

    st.subheader("Weekly Sales with Detected Anomalies")

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        weekly_sales["Order Date"],
        weekly_sales["Sales"],
        color="blue",
        linewidth=2,
        label="Weekly Sales"
    )

    ax.scatter(
        anomalies["Order Date"],
        anomalies["Sales"],
        color="red",
        marker="o",
        s=100,
        label="Anomalies"
    )

    # Display Chart 
    ax.set_title("Isolation Forest Anomaly Detection")
    ax.set_xlabel("Order Date")
    ax.set_ylabel("Weekly Sales")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    st.divider()

    st.subheader("Detected Anomaly Weeks")

    st.dataframe(
        anomalies[
            [
                "Order Date",
                "Sales"
            ]
        ],
        width="stretch"
    )

    st.metric(
        "Total Anomalies",
        len(anomalies)
    )

    st.info(
        """
Isolation Forest detected several unusual sales weeks that differ from the normal sales pattern.
These anomalies may represent:
• Festive season sales
• Promotional campaigns
• Supply chain disruptions
• Sudden changes in customer demand
"""
    )

elif page == "Product Demand Segments":

    st.subheader("Demand Segmentation (PCA Visualization)")

    fig, ax = plt.subplots(figsize=(10,6))

    scatter = ax.scatter(
        subcategory_features["PCA1"],
        subcategory_features["PCA2"],
        c=subcategory_features["Cluster"],
        cmap="viridis",
        s=120
    )

    for i in range(len(subcategory_features)):
        ax.text(
            subcategory_features["PCA1"].iloc[i],
            subcategory_features["PCA2"].iloc[i],
            subcategory_features["Sub-Category"].iloc[i],
            fontsize=8
        )

    ax.set_title("Product Demand Segments")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

    plt.colorbar(scatter)

    st.pyplot(fig)
    st.subheader("Product Demand Segments")

    st.dataframe(

        subcategory_features[
            [
                "Sub-Category",
                "Cluster",
                "Demand_Segment"
            ]
        ],
        width="stretch"
    )

    st.subheader("Products in Each Cluster")

    cluster_count = (
        subcategory_features["Demand_Segment"]
        .value_counts()
    )

    st.bar_chart(cluster_count)
    st.info(
        """
    The clustering model groups products based on their sales volume,
    growth rate, volatility, and average order value.

    These demand segments help businesses optimize inventory,
    reduce storage costs, and improve stock planning.
    """
    )
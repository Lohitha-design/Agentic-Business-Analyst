import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from main import main
from src.Injestion.dataLoading import load_file

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Agentic Business Analyst",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>

html, body, [class*="css"]{
    background-color:#1e1e1e;
    color:#f5f5f5;
    font-family:Georgia, serif;
}

.stApp{
    background:#1e1e1e;
}

.title{
    text-align:center;
    font-size:3.5rem;
    color:#0f8f63;
    font-weight:bold;
    margin-bottom:25px;
}

.card{
    background:#12372a;
    border:2px solid #0f8f63;
    padding:20px;
    margin-top:20px;
}

.metric{
    background:#12372a;
    border:2px solid #0f8f63;
    padding:15px;
    text-align:center;
}

.metric h2{
    color:white;
    margin:0;
}

.metric p{
    color:#d8d8d8;
}

.stButton > button{
    width:100%;
    background:#0f8f63;
    color:white;
    border:none;
    padding:12px;
    font-weight:bold;
}

.stButton > button:hover{
    background:#146c4f;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown(
    "<div class='title'>Agentic Business Analyst</div>",
    unsafe_allow_html=True
)

# =====================================================
# FILE UPLOAD
# =====================================================
uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=["csv", "xlsx", "json"]
)

# =====================================================
# MAIN
# =====================================================
if uploaded_file:

    df = load_file(uploaded_file)

    # =================================================
    # METRICS
    # =================================================
    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        ("Rows", df.shape[0]),
        ("Columns", df.shape[1]),
        ("Duplicates", df.duplicated().sum()),
        ("Missing Values", df.isnull().sum().sum())
    ]

    for col, metric in zip([c1, c2, c3, c4], metrics):

        with col:
            st.markdown(f"""
            <div class='metric'>
                <p>{metric[0]}</p>
                <h2>{metric[1]}</h2>
            </div>
            """, unsafe_allow_html=True)

    # =================================================
    # DATA PREVIEW
    # =================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # =================================================
    # HEATMAP
    # =================================================
    numeric_df = df.select_dtypes(include="number")

    if not numeric_df.empty:

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.subheader("Correlation Heatmap")

        fig, ax = plt.subplots(figsize=(12,6))

        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap="Greens",
            linewidths=0.5,
            ax=ax
        )

        st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)

    # =================================================
    # COLUMN VISUALIZATION
    # =================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("Column Visualization")

    column = st.selectbox(
        "Select Column",
        df.columns
    )

    if pd.api.types.is_numeric_dtype(df[column]):
        st.line_chart(df[column])
    else:
        st.bar_chart(df[column].value_counts())

    st.markdown("</div>", unsafe_allow_html=True)

    # =================================================
    # TARGET
    # =================================================
    target = st.selectbox(
        "Select Target Variable",
        numeric_df.columns
    )

    # =================================================
    # GENERATE STRATEGY
    # =================================================
    if st.button("Generate Strategy"):

        with st.spinner(
            "Running strategic intelligence pipeline..."
        ):

            try:

                result = main(df, target)

                st.markdown("<div class='card'>", unsafe_allow_html=True)

                st.subheader("Generated Strategy")

                st.write(result)

                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:

                st.error(f"Error: {e}")
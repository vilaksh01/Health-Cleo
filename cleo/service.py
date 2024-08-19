import streamlit as st
import os
from fit import FoodIntoleranceAnalysisService
import hashlib
import pandas as pd

# Initialize the service
@st.cache_resource
def get_service():
    upstage_api_key = os.getenv("UPSTAGE_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")

    if not upstage_api_key or not tavily_api_key:
        st.error("API keys are missing! Please set them as environment variables.")
        st.stop()
        
    return FoodIntoleranceAnalysisService(upstage_api_key=upstage_api_key, tavily_api_key= tavily_api_key)

service = get_service()

st.title("Food Intolerance Analysis")

# Function to compute file hash
def compute_file_hash(file_content):
    return hashlib.md5(file_content).hexdigest()

# Sidebar for uploading PDF and selecting analysis type
with st.sidebar:
    st.header("Upload and Analyze")
    uploaded_file = st.file_uploader("Upload Food Intolerance Report (PDF)", type="pdf")
    analysis_type = st.radio("Select Analysis Type", ["Text Input", "Image Upload"])

# Main content area
if uploaded_file:
    file_hash = compute_file_hash(uploaded_file.getvalue())
    
    # Check if we've already processed this file
    if 'processed_file_hash' not in st.session_state or st.session_state.processed_file_hash != file_hash:
        with st.spinner("Processing PDF..."):
            pdf_path = f"/tmp/{uploaded_file.name}"
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            report = service.process_pdf(pdf_path)
            st.session_state.report = report
            st.session_state.processed_file_hash = file_hash
        st.success("PDF processed successfully!")
    else:
        st.info("Using previously processed PDF results.")
        report = st.session_state.report

    st.subheader("Reference Range")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Elevated", report.reference_range.elevated)
    with col2:
        st.metric("Borderline", report.reference_range.borderline)
    with col3:
        st.metric("Normal", report.reference_range.normal)

    st.subheader("Food Items")
    food_items_df = pd.DataFrame([
        {"Name": item.name, "Value (U/mL)": item.value, "Category": item.category}
        for item in report.food_items
    ])
    st.dataframe(food_items_df)

if analysis_type == "Text Input":
    product_name = st.text_input("Enter product name")
    if st.button("Analyze Product"):
        with st.spinner("Analyzing product..."):
            analysis = service.analyze_product_from_text(product_name)
        st.subheader(f"Analysis for {analysis.product_name}")
        st.write(f"**Ingredients:** {', '.join(analysis.ingredients)}")
        
        st.write("**Suitability:**")
        suitability_df = pd.DataFrame([
            {"Ingredient": ingredient, "Suitability": suitability}
            for ingredient, suitability in analysis.suitability.items()
        ])
        st.dataframe(suitability_df)
        
        st.metric("Overall Rating", analysis.overall_rating)
        st.write(f"**Explanation:** {analysis.explanation}")

elif analysis_type == "Image Upload":
    uploaded_image = st.file_uploader("Upload product image", type=["jpg", "jpeg", "png"])
    if uploaded_image and st.button("Analyze Image"):
        with st.spinner("Analyzing image..."):
            image_path = f"/tmp/{uploaded_image.name}"
            with open(image_path, "wb") as f:
                f.write(uploaded_image.getvalue())
            analysis = service.analyze_product_from_image(image_path)
        st.subheader(f"Analysis for {analysis.product_name}")
        st.write(f"**Ingredients:** {', '.join(analysis.ingredients)}")
        
        st.write("**Suitability:**")
        suitability_df = pd.DataFrame([
            {"Ingredient": ingredient, "Suitability": suitability}
            for ingredient, suitability in analysis.suitability.items()
        ])
        st.dataframe(suitability_df)
        
        st.metric("Overall Rating", analysis.overall_rating)
        st.write(f"**Explanation:** {analysis.explanation}")

# Chat interface
st.subheader("Chat with Food Intolerance Assistant")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about food intolerances"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in service.llm.stream(prompt):
            if isinstance(response, dict) and 'content' in response:
                chunk = response['content']
            elif isinstance(response, str):
                chunk = response
            else:
                continue
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
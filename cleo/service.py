import streamlit as st
import os
from fit import FoodIntoleranceAnalysisService

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

# Sidebar for uploading PDF and selecting analysis type
with st.sidebar:
    st.header("Upload and Analyze")
    uploaded_file = st.file_uploader("Upload Food Intolerance Report (PDF)", type="pdf")
    analysis_type = st.radio("Select Analysis Type", ["Text Input", "Image Upload"])

# Main content area
if uploaded_file:
    with st.spinner("Processing PDF..."):
        pdf_path = f"/tmp/{uploaded_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        report = service.process_pdf(pdf_path)
    
    st.success("PDF processed successfully!")
    st.subheader("Reference Range")
    st.write(f"Elevated: {report.reference_range.elevated}")
    st.write(f"Borderline: {report.reference_range.borderline}")
    st.write(f"Normal: {report.reference_range.normal}")
    
    st.subheader("Food Items")
    for item in report.food_items:
        st.write(f"{item.name}: {item.value} U/mL - {item.category}")

if analysis_type == "Text Input":
    product_name = st.text_input("Enter product name")
    if st.button("Analyze Product"):
        with st.spinner("Analyzing product..."):
            analysis = service.analyze_product_from_text(product_name)
        st.subheader(f"Analysis for {analysis.product_name}")
        st.write(f"Ingredients: {', '.join(analysis.ingredients)}")
        st.write("Suitability:")
        for ingredient, suitability in analysis.suitability.items():
            st.write(f"  {ingredient}: {suitability}")
        st.write(f"Overall Rating: {analysis.overall_rating}")
        st.write(f"Explanation: {analysis.explanation}")

elif analysis_type == "Image Upload":
    uploaded_image = st.file_uploader("Upload product image", type=["jpg", "jpeg", "png"])
    if uploaded_image and st.button("Analyze Image"):
        with st.spinner("Analyzing image..."):
            image_path = f"/tmp/{uploaded_image.name}"
            with open(image_path, "wb") as f:
                f.write(uploaded_image.getvalue())
            analysis = service.analyze_product_from_image(image_path)
        st.subheader(f"Analysis for {analysis.product_name}")
        st.write(f"Ingredients: {', '.join(analysis.ingredients)}")
        st.write("Suitability:")
        for ingredient, suitability in analysis.suitability.items():
            st.write(f"  {ingredient}: {suitability}")
        st.write(f"Overall Rating: {analysis.overall_rating}")
        st.write(f"Explanation: {analysis.explanation}")

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
            full_response += str(response)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

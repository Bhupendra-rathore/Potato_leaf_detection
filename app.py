# Potato Leaf Disease Detection - Streamlit App
# Save this as 'app.py' and run with: streamlit run app.py

import streamlit as st
import pickle
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow import keras
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from io import BytesIO
import base64

# Page configuration
st.set_page_config(
    page_title="🥔 Potato Disease Detector",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4682B4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 20px 0;
    }
    .healthy-box {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 20px 0;
    }
    .diseased-box {
        background: linear-gradient(90deg, #ff6b6b 0%, #feca57 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 20px 0;
    }
    .info-box {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class PotatoDiseasePredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.class_names = None
        self.img_height = 224
        self.img_width = 224
        self.load_model()
    
    @st.cache_resource
    def load_model(_self):
        """Load the trained model from pickle file"""
        try:
            with open(_self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            _self.model = model_data['model']
            _self.class_names = model_data['class_names']
            _self.img_height = model_data['img_height']
            _self.img_width = model_data['img_width']
            
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        # Resize image
        image = image.resize((self.img_width, self.img_height))
        
        # Convert to array and normalize
        img_array = np.array(image) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image):
        """Make prediction on image"""
        if self.model is None:
            return None, None, None
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Make prediction
        predictions = self.model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        
        return predicted_class, confidence, predictions[0]

def create_confidence_chart(predictions, class_names):
    """Create confidence chart"""
    df = pd.DataFrame({
        'Class': class_names,
        'Confidence': predictions * 100
    })
    
    fig = px.bar(df, x='Class', y='Confidence', 
                 title='Prediction Confidence by Class',
                 color='Confidence',
                 color_continuous_scale='Viridis')
    
    fig.update_layout(
        xaxis_title="Disease Class",
        yaxis_title="Confidence (%)",
        showlegend=False,
        height=400
    )
    
    return fig

def create_pie_chart(predictions, class_names):
    """Create pie chart for predictions"""
    fig = go.Figure(data=[go.Pie(
        labels=class_names,
        values=predictions,
        hole=.3,
        textinfo='label+percent',
        textfont_size=12,
        marker=dict(colors=['#ff9999', '#66b3ff', '#99ff99'])
    )])
    
    fig.update_layout(
        title="Prediction Distribution",
        annotations=[dict(text='Confidence', x=0.5, y=0.5, font_size=16, showarrow=False)],
        height=400
    )
    
    return fig

def get_disease_info(predicted_class):
    """Get information about the predicted disease"""
    disease_info = {
        'Healthy': {
            'description': "🌿 The potato plant appears to be healthy with no signs of disease.",
            'recommendations': [
                "Continue regular monitoring",
                "Maintain proper irrigation",
                "Ensure adequate nutrition",
                "Practice crop rotation"
            ],
            'color': 'healthy-box'
        },
        'Early': {
            'description': "⚠️ Early Blight detected. This is a fungal disease caused by Alternaria solani.",
            'symptoms': [
                "Dark spots on leaves with concentric rings",
                "Yellowing around spots",
                "Premature leaf drop",
                "Reduced yield potential"
            ],
            'recommendations': [
                "Apply fungicide immediately",
                "Improve air circulation",
                "Remove infected plant debris",
                "Avoid overhead watering",
                "Consider resistant varieties"
            ],
            'color': 'diseased-box'
        },
        'Infected': {
            'description': "🚨 Late Blight detected. This is a serious disease caused by Phytophthora infestans.",
            'symptoms': [
                "Water-soaked spots on leaves",
                "White fuzzy growth on leaf undersides",
                "Rapid spread in cool, wet conditions",
                "Can destroy entire crops quickly"
            ],
            'recommendations': [
                "Apply fungicide immediately",
                "Remove and destroy infected plants",
                "Improve drainage",
                "Increase plant spacing",
                "Consider emergency harvest if severe"
            ],
            'color': 'diseased-box'
        }
    }
    
    return disease_info.get(predicted_class, disease_info['Healthy'])

def main():
    # Header
    st.markdown('<h1 class="main-header">🥔 Potato Disease Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Plant Health Analysis</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📋 Navigation")
    page = st.sidebar.selectbox("Choose a page:", ["🏠 Disease Detection", "📊 Model Info", "❓ Help"])
    
    # Load model
    model_path = "potato_disease_model.pkl"  # Update this path if needed
    
    try:
        predictor = PotatoDiseasePredictor(model_path)
        model_loaded = True
    except:
        model_loaded = False
        st.error("❌ Model file not found! Please ensure 'potato_disease_model.pkl' is in the same directory.")
        st.info("Train the model first using the Jupyter notebook, then run this Streamlit app.")
        return
    
    if page == "🏠 Disease Detection":
        st.header("Upload Potato Leaf Image")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a potato leaf for disease detection"
        )
        
        # Camera input option
        st.subheader("Or take a photo:")
        camera_image = st.camera_input("Take a picture of the potato leaf")
        
        # Use either uploaded file or camera image
        image_source = uploaded_file if uploaded_file else camera_image
        
        if image_source is not None:
            # Create columns for layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📸 Uploaded Image")
                image = Image.open(image_source)
                st.image(image, caption="Input Image", use_column_width=True)
                
                # Image info
                st.write(f"**Image Size:** {image.size}")
                st.write(f"**Image Mode:** {image.mode}")
            
            with col2:
                if st.button("🔍 Analyze Image", type="primary"):
                    with st.spinner("Analyzing image... Please wait"):
                        # Make prediction
                        predicted_class, confidence, all_predictions = predictor.predict(image)
                        
                        if predicted_class:
                            # Display prediction
                            disease_info = get_disease_info(predicted_class)
                            
                            if predicted_class == 'Healthy':
                                st.markdown(f'''
                                <div class="{disease_info['color']}">
                                    <h3>✅ Prediction: {predicted_class}</h3>
                                    <h4>Confidence: {confidence:.2%}</h4>
                                </div>
                                ''', unsafe_allow_html=True)
                            else:
                                st.markdown(f'''
                                <div class="{disease_info['color']}">
                                    <h3>⚠️ Prediction: {predicted_class} Blight</h3>
                                    <h4>Confidence: {confidence:.2%}</h4>
                                </div>
                                ''', unsafe_allow_html=True)
                            
                            # Disease information
                            st.subheader("📋 Analysis Results")
                            st.markdown(f'<div class="info-box">{disease_info["description"]}</div>', 
                                      unsafe_allow_html=True)
                            
                            if 'symptoms' in disease_info:
                                st.subheader("🔍 Symptoms")
                                for symptom in disease_info['symptoms']:
                                    st.write(f"• {symptom}")
                            
                            st.subheader("💡 Recommendations")
                            for rec in disease_info['recommendations']:
                                st.write(f"✓ {rec}")
                            
                            # Visualization
                            st.subheader("📊 Prediction Confidence")
                            
                            # Confidence chart
                            fig1 = create_confidence_chart(all_predictions, predictor.class_names)
                            st.plotly_chart(fig1, use_container_width=True)
                            
                            # Pie chart
                            fig2 = create_pie_chart(all_predictions, predictor.class_names)
                            st.plotly_chart(fig2, use_container_width=True)
                            
                            # Detailed predictions
                            st.subheader("📈 Detailed Predictions")
                            pred_df = pd.DataFrame({
                                'Class': predictor.class_names,
                                'Confidence': [f"{pred:.2%}" for pred in all_predictions],
                                'Probability': all_predictions
                            }).sort_values('Probability', ascending=False)
                            
                            st.dataframe(pred_df[['Class', 'Confidence']], use_container_width=True)
    
    elif page == "📊 Model Info":
        st.header("Model Information")
        
        if model_loaded:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔧 Model Details")
                st.write(f"**Classes:** {', '.join(predictor.class_names)}")
                st.write(f"**Input Size:** {predictor.img_width} x {predictor.img_height}")
                st.write(f"**Model Type:** Convolutional Neural Network (CNN)")
                
            with col2:
                st.subheader("📋 Class Information")
                for i, class_name in enumerate(predictor.class_names):
                    st.write(f"**{i}.** {class_name}")
            
            # Model architecture (if available)
            if hasattr(predictor.model, 'summary'):
                st.subheader("🏗️ Model Architecture")
                # Create a string buffer to capture the model summary
                from io import StringIO
                import sys
                
                old_stdout = sys.stdout
                sys.stdout = buffer = StringIO()
                
                predictor.model.summary()
                model_summary = buffer.getvalue()
                sys.stdout = old_stdout
                
                st.text(model_summary)
        else:
            st.error("Model not loaded properly.")
    
    elif page == "❓ Help":
        st.header("Help & Instructions")
        
        st.subheader("🚀 How to Use")
        st.write("""
        1. **Upload Image**: Go to the 'Disease Detection' page and upload a clear image of a potato leaf
        2. **Take Photo**: Alternatively, use your camera to take a photo directly
        3. **Analyze**: Click the 'Analyze Image' button to get predictions
        4. **Review Results**: Check the disease classification, confidence scores, and recommendations
        """)
        
        st.subheader("📷 Image Requirements")
        st.write("""
        - **Format**: JPG, JPEG, or PNG
        - **Quality**: Clear, well-lit images work best
        - **Subject**: Focus on the potato leaf, avoid too much background
        - **Size**: Any size (will be automatically resized)
        """)
        
        st.subheader("🔬 Disease Classes")
        disease_descriptions = {
            "Healthy": "Normal, disease-free potato leaves",
            "Early Blight": "Fungal disease with dark spots and concentric rings",
            "Late Blight": "Serious disease with water-soaked spots and white growth"
        }
        
        for disease, description in disease_descriptions.items():
            st.write(f"**{disease}**: {description}")
        
        st.subheader("⚠️ Important Notes")
        st.warning("""
        - This AI model is for assistance only and should not replace professional agricultural advice
        - For severe infections, consult with agricultural experts
        - Early detection and treatment are key to preventing crop loss
        """)
        
        st.subheader("🛠️ Technical Requirements")
        st.info("""
        - Ensure 'potato_disease_model.pkl' is in the app directory
        - The model was trained on specific potato disease images
        - Best results with similar lighting and image conditions
        """)
    
    # Footer
    st.sidebar

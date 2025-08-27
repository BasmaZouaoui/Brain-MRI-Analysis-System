import streamlit as st
import torch
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTForImageClassification
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
from torch import nn

# Set page config
st.set_page_config(
    page_title="Brain MRI Analysis System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Attention Rollout for Vision Transformer
class AttentionRollout:
    def __init__(self, model, discard_ratio=0.9, head_fusion="mean"):
        self.model = model
        self.discard_ratio = discard_ratio
        self.head_fusion = head_fusion
        
    def get_attention_maps(self, input_tensor):
        """Extract attention maps from ViT model"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor, output_attentions=True)
            attentions = outputs.attentions  # List of attention tensors from each layer
            
        # Average attention heads
        attention_maps = []
        for attention in attentions:
            if self.head_fusion == "mean":
                attention_map = attention.mean(dim=1)  # Average over attention heads
            elif self.head_fusion == "max":
                attention_map = attention.max(dim=1)[0]
            else:
                attention_map = attention[:, 0]  # Use first head
            attention_maps.append(attention_map)
        
        return attention_maps
    
    def generate_rollout(self, input_tensor, class_idx=None):
        """Generate attention rollout visualization"""
        attention_maps = self.get_attention_maps(input_tensor)
        
        # Start with identity matrix
        rollout = torch.eye(attention_maps[0].size(-1))
        
        # Apply rollout through layers
        for attention in attention_maps:
            # Add residual connection (identity matrix)
            attention_head_avg = attention[0]  # Take first batch item
            attention_head_avg = attention_head_avg + torch.eye(attention_head_avg.size(-1))
            attention_head_avg = attention_head_avg / attention_head_avg.sum(dim=-1, keepdim=True)
            
            # Matrix multiply for rollout
            rollout = torch.matmul(attention_head_avg, rollout)
        
        # Get attention for classification token (index 0) to all patches
        mask = rollout[0, 1:]  # Exclude CLS token, get attention to patches
        
        # Reshape to spatial dimensions (14x14 for 224x224 input with patch size 16)
        width = height = int(np.sqrt(len(mask)))
        mask = mask.reshape(height, width)
        
        # Normalize
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        
        return mask.numpy()

# Medical information dictionary
MEDICAL_INFO = {
    'glioma': {
        'name': 'Glioma',
        'description': 'A type of brain tumor that originates from glial cells. Gliomas can vary in aggressiveness and are classified into different grades.',
        'symptoms': [
            'Persistent headaches, often worse in the morning',
            'Seizures (new onset)',
            'Progressive neurological deficits',
            'Nausea and vomiting',
            'Vision or speech problems',
            'Personality or cognitive changes'
        ],
        'urgency': 'HIGH',
        'urgency_color': '#dc3545',
        'next_steps': [
            'Immediate referral to neurosurgery',
            'Consider MRI with contrast if not already done',
            'Neurological examination',
            'Discuss treatment options with oncology team'
        ],
        'treatment_summary': 'Treatment typically involves surgery, radiation therapy, and/or chemotherapy depending on tumor grade and location.'
    },
    'meningioma': {
        'name': 'Meningioma',
        'description': 'A tumor arising from the meninges (protective layers around the brain). Most meningiomas are benign and slow-growing.',
        'symptoms': [
            'Gradual onset headaches',
            'Vision changes or double vision',
            'Hearing problems or ringing in ears',
            'Weakness in arms or legs',
            'Memory problems',
            'Seizures (less common)'
        ],
        'urgency': 'MODERATE',
        'urgency_color': '#fd7e14',
        'next_steps': [
            'Neurosurgery consultation for evaluation',
            'Monitor with serial imaging if asymptomatic',
            'Assess for mass effect symptoms',
            'Consider hormone therapy evaluation'
        ],
        'treatment_summary': 'Treatment depends on size and symptoms. Options include observation, surgery, or radiation therapy.'
    },
    'pituitary': {
        'name': 'Pituitary Adenoma',
        'description': 'A tumor of the pituitary gland that can affect hormone production and cause mass effect symptoms.',
        'symptoms': [
            'Visual field defects (especially peripheral vision)',
            'Hormonal symptoms (varies by tumor type)',
            'Headaches',
            'Fatigue and weakness',
            'Sexual dysfunction',
            'Growth problems (in children)'
        ],
        'urgency': 'MODERATE',
        'urgency_color': '#fd7e14',
        'next_steps': [
            'Endocrinology referral for hormone evaluation',
            'Ophthalmology referral for visual field testing',
            'Consider neurosurgery consultation',
            'Complete hormonal workup'
        ],
        'treatment_summary': 'Treatment may include medication, surgery (transsphenoidal), or radiation therapy based on tumor type and symptoms.'
    },
    'notumor': {
        'name': 'No Tumor Detected',
        'description': 'The analysis suggests normal brain tissue without evidence of tumor. This is a reassuring finding.',
        'symptoms': [],
        'urgency': 'LOW',
        'urgency_color': '#28a745',
        'next_steps': [
            'Continue routine clinical evaluation',
            'Address presenting symptoms with appropriate workup',
            'Consider other differential diagnoses if symptoms persist',
            'Routine follow-up as clinically indicated'
        ],
        'treatment_summary': 'No tumor-specific treatment needed. Focus on addressing presenting symptoms through appropriate clinical evaluation.'
    }
}

@st.cache_resource
def load_vit_model(model_name_or_path):
    """Load the trained ViT model from Hugging Face"""
    try:
        # Load processor and model
        processor = ViTImageProcessor.from_pretrained(model_name_or_path)
        model = ViTForImageClassification.from_pretrained(model_name_or_path)
        
        # Set to evaluation mode
        model.eval()
        
        return processor, model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

class BrainMRIAnalyzer:
    def __init__(self, model_name_or_path="basmazouaoui/vit-brain-tumor-classifier"):
        self.model_name_or_path = model_name_or_path
        self.class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.processor, self.model = load_vit_model(model_name_or_path)
        
        # Initialize attention rollout if model is loaded
        if self.model is not None:
            self.attention_rollout = AttentionRollout(self.model)
    
    def preprocess_image(self, image):
        """Preprocess uploaded image for ViT model prediction"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Use the ViT processor to handle preprocessing
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs, image
    
    def analyze_image(self, inputs):
        """Analyze the processed image and return predictions"""
        if self.model is None:
            return None, None, None
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = F.softmax(outputs.logits, dim=-1)
            
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = torch.max(predictions).item() * 100
        all_predictions = predictions[0].numpy()
        
        return predicted_class, confidence, all_predictions
    
    def generate_attention_map(self, inputs, predicted_class=None):
        """Generate attention map showing what the ViT model focused on"""
        try:
            if self.model is None:
                return None
                
            # Generate attention rollout
            attention_map = self.attention_rollout.generate_rollout(inputs['pixel_values'])
            return attention_map
        except Exception as e:
            st.warning(f"Could not generate attention map: {str(e)}")
            return None
    
    def create_overlay_image(self, original_image, attention_map, img_size=(224, 224)):
        """Create overlay of attention map on original image"""
        try:
            # Resize original image
            resized_image = original_image.resize(img_size)
            img_array = np.array(resized_image)
            
            # Resize attention map to match image
            attention_resized = cv2.resize(attention_map, img_size)
            
            # Create colored heatmap
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * attention_resized), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Create overlay
            overlay = cv2.addWeighted(img_array, 0.7, heatmap_colored, 0.3, 0)
            return overlay
        except Exception as e:
            st.warning(f"Could not create overlay image: {str(e)}")
            return original_image

def main():
    # Custom CSS for medical professional interface
    st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        color: #1e3a8a;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    .confidence-score {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .urgency-high { 
        background: #dc3545; 
        color: white; 
        padding: 0.5rem 1rem; 
        border-radius: 20px; 
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .urgency-moderate { 
        background: #fd7e14; 
        color: white; 
        padding: 0.5rem 1rem; 
        border-radius: 20px; 
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .urgency-low { 
        background: #28a745; 
        color: white; 
        padding: 0.5rem 1rem; 
        border-radius: 20px; 
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .medical-section {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Model configuration
    st.sidebar.header("🔧 Model Configuration")
    model_name = st.sidebar.text_input(
        "Hugging Face Model Path",
        value="basmazouaoui/vit-brain-tumor-classifier",
        help="Enter your Hugging Face model repository path"
    )
    
    # Initialize analyzer with custom model path
    analyzer = BrainMRIAnalyzer(model_name)
    
    # Header
    st.markdown('<div class="main-title">🧠 Brain MRI Analysis System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-Powered Brain Tumor Detection using Vision Transformer</div>', unsafe_allow_html=True)
    
    # Check if model is loaded
    if analyzer.model is None or analyzer.processor is None:
        st.error(f"🚫 **System Error:** Could not load model '{model_name}' from Hugging Face.")
        st.info("""
        **Troubleshooting:**
        1. Ensure your model is public or you're authenticated with Hugging Face
        2. Verify the model path is correct (format: username/model-name)
        3. Check your internet connection
        4. Make sure the model contains the required ViT architecture
        """)
        return
    
    st.success("✅ **System Status:** Vision Transformer Model Ready")
    
    # Model info
    with st.expander("📋 Model Information"):
        st.write(f"**Model Path:** {model_name}")
        st.write(f"**Architecture:** Vision Transformer (ViT)")
        st.write(f"**Classes:** {', '.join(analyzer.class_names)}")
        st.write(f"**Processor:** {type(analyzer.processor).__name__}")
    
    # File upload section
    st.markdown("---")
    st.header("📋 Patient MRI Upload")
    
    uploaded_file = st.file_uploader(
        "Upload Brain MRI Scan",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Supported formats: JPEG, PNG, BMP, TIFF • Recommended: T1/T2 weighted MRI images"
    )
    
    if uploaded_file is not None:
        # Process uploaded image
        image = Image.open(uploaded_file)
        
        # Analysis section
        st.markdown("---")
        st.header("🔬 Analysis Results")
        
        with st.spinner("🧠 Vision Transformer analysis in progress..."):
            inputs, processed_image = analyzer.preprocess_image(image)
            predicted_class, confidence, all_predictions = analyzer.analyze_image(inputs)
        
        if predicted_class is not None:
            predicted_condition = analyzer.class_names[predicted_class]
            medical_info = MEDICAL_INFO[predicted_condition]
            
            # Main results display
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("📸 MRI Scan")
                st.image(image, caption=f"File: {uploaded_file.name}", use_container_width=True)
                
                # Quick stats
                st.write("**Scan Information:**")
                st.write(f"• Resolution: {image.size[0]}×{image.size[1]}")
                st.write(f"• Analysis Time: {datetime.now().strftime('%H:%M:%S')}")
                st.write(f"• Image Format: {image.format if image.format else 'Unknown'}")
                st.write(f"• Model: Vision Transformer")
            
            with col2:
                # Primary diagnosis
                st.markdown(f"""
                <div class="result-card">
                    <h2>🎯 Primary Finding</h2>
                    <h1>{medical_info['name']}</h1>
                    <div class="confidence-score">{confidence:.1f}% Confidence</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Urgency level
                urgency_class = f"urgency-{medical_info['urgency'].lower()}"
                st.markdown(f"""
                <div style="text-align: center; margin: 1rem 0;">
                    <span class="{urgency_class}">🚨 {medical_info['urgency']} PRIORITY</span>
                </div>
                """, unsafe_allow_html=True)
        
            # Detailed probability breakdown
            st.subheader("📊 Diagnostic Confidence Levels")
            
            prob_data = []
            for i, class_name in enumerate(analyzer.class_names):
                display_name = MEDICAL_INFO[class_name]['name']
                probability = all_predictions[i] * 100
                prob_data.append({
                    'Condition': display_name,
                    'Probability': probability,
                    'Class': class_name
                })
            
            prob_df = pd.DataFrame(prob_data)
            prob_df = prob_df.sort_values('Probability', ascending=True)
            
            # Horizontal bar chart
            fig = px.bar(
                prob_df, 
                x='Probability', 
                y='Condition',
                orientation='h',
                color='Probability',
                color_continuous_scale=['#fee2e2', '#dc2626'],
                title="Vision Transformer Confidence Levels by Condition (%)"
            )
            fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
            fig.update_layout(
                height=300,
                showlegend=False,
                xaxis_title="Confidence Level (%)",
                yaxis_title="",
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Attention Visualization
            st.subheader("🔍 Vision Transformer Attention")
            st.write("This shows which image patches the Vision Transformer focused on during classification:")
            
            with st.spinner("Generating attention visualization..."):
                attention_map = analyzer.generate_attention_map(inputs, predicted_class)
            
            if attention_map is not None:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Original MRI**")
                    st.image(processed_image, use_container_width=True)
                
                with col2:
                    st.write("**Attention Map**")
                    fig_att, ax_att = plt.subplots(figsize=(5, 5))
                    im = ax_att.imshow(attention_map, cmap='hot', alpha=0.8)
                    ax_att.axis('off')
                    ax_att.set_title("High Attention = Red")
                    plt.colorbar(im, ax=ax_att, shrink=0.8)
                    st.pyplot(fig_att)
                    plt.close(fig_att)
                
                with col3:
                    st.write("**Overlay View**")
                    overlay = analyzer.create_overlay_image(processed_image, attention_map)
                    st.image(overlay, use_container_width=True)
            
            # Medical Information
            st.markdown("---")
            st.header("🏥 Clinical Information")
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown(f"""
                <div class="medical-section">
                    <h3>📋 Clinical Description</h3>
                    <p>{medical_info['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if medical_info['symptoms']:
                    st.markdown(f"""
                    <div class="medical-section">
                        <h3>⚕️ Associated Symptoms</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for symptom in medical_info['symptoms']:
                        st.write(f"• {symptom}")
            
            with col2:
                st.markdown(f"""
                <div class="medical-section">
                    <h3>📋 Recommended Actions</h3>
                </div>
                """, unsafe_allow_html=True)
                
                for i, action in enumerate(medical_info['next_steps'], 1):
                    st.write(f"{i}. {action}")
                
                st.markdown(f"""
                <div class="medical-section">
                    <h3>💊 Treatment Overview</h3>
                    <p>{medical_info['treatment_summary']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Generate Report
            st.markdown("---")
            st.header("📄 Clinical Report")
            
            if st.button("📋 Generate Full Clinical Report", type="primary"):
                report_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                report = f"""
BRAIN MRI AI ANALYSIS REPORT
============================

PATIENT INFORMATION:
-------------------
Image File: {uploaded_file.name}
Analysis Date: {report_time}
Analysis Method: Vision Transformer (ViT)
Model: {model_name}

FINDINGS:
---------
Primary Finding: {medical_info['name']}
AI Confidence: {confidence:.1f}%
Priority Level: {medical_info['urgency']}

DETAILED PROBABILITIES:
----------------------
"""
                for _, row in prob_df.iterrows():
                    report += f"{row['Condition']}: {row['Probability']:.1f}%\n"
                
                report += f"""

CLINICAL DESCRIPTION:
--------------------
{medical_info['description']}

RECOMMENDED NEXT STEPS:
----------------------
"""
                for i, action in enumerate(medical_info['next_steps'], 1):
                    report += f"{i}. {action}\n"
                
                report += f"""

TREATMENT SUMMARY:
-----------------
{medical_info['treatment_summary']}

TECHNICAL DETAILS:
-----------------
Model Architecture: Vision Transformer (ViT)
Attention Mechanism: Multi-head self-attention
Input Resolution: 224x224 pixels
Patch Size: 16x16 pixels
Model Source: Hugging Face Hub

IMPORTANT DISCLAIMER:
--------------------
This AI analysis is intended to assist healthcare professionals and should not replace clinical judgment. 
All findings should be correlated with clinical presentation and confirmed through appropriate medical evaluation.
The Vision Transformer model has been trained on medical imaging data but may not account for all clinical variables.

Report generated by Brain MRI AI Analysis System (ViT-based)
"""
                
                st.text_area("📋 Clinical Report", report, height=400)
                
                # Download button
                st.download_button(
                    label="💾 Download Report",
                    data=report,
                    file_name=f"ViT_MRI_Analysis_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        else:
            st.error("❌ Analysis failed. Please try uploading a different image.")
        
        # Important disclaimers
        st.markdown("---")
        st.error("""
        ⚠️ **IMPORTANT MEDICAL DISCLAIMER**
        
        This Vision Transformer AI system is designed to ASSIST medical professionals, not replace clinical judgment. 
        
        • Always correlate AI findings with clinical presentation
                 
        • Confirm findings through appropriate medical evaluation  
                 
        • This tool is for educational and assistive purposes only
                 
        • Final diagnosis and treatment decisions remain with the healthcare provider
        """)
    
    else:
        # Instructions when no image uploaded
        st.info("👆 **Instructions:** Upload a brain MRI scan above to begin Vision Transformer analysis")
        
        # Show what the system can detect
        st.subheader("🎯 Conditions This System Can Detect")
        
        detection_cols = st.columns(2)
        
        with detection_cols[0]:
            st.markdown("""
            **🔴 Glioma**
            - Aggressive brain tumors
            - Requires urgent attention
            - Most common primary brain tumor
            
            **🟡 Meningioma**  
            - Usually benign tumors
            - Slow growing
            - Often manageable with monitoring
            """)
        
        with detection_cols[1]:
            st.markdown("""
            **🔵 Pituitary Adenoma**
            - Affects hormone production
            - May cause vision problems  
            - Treatable with various approaches
            
            **🟢 No Tumor**
            - Normal healthy brain tissue
            - Reassuring finding
            - Rules out tumor pathology
            """)

if __name__ == "__main__":
    main()
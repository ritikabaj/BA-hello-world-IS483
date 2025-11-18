import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="SMU Wellness Attendance Predictor",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .high-prob {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .medium-prob {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
    }
    .low-prob {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🎓 SMU Wellness Attendance Predictor</p>', unsafe_allow_html=True)
st.markdown("---")

# Load model function
@st.cache_resource
def load_model():
    try:
        with open('attendance_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("⚠️ Model file 'attendance_model.pkl' not found. Please train the model first.")
        return None

# Initialize model
model_data = load_model()

if model_data is not None:
    model = model_data['model']
    preprocessor = model_data['preprocessor']
    
    # Sidebar - Model Info
    with st.sidebar:
        st.header("📊 Model Information")
        if 'metrics' in model_data:
            metrics = model_data['metrics']
            st.metric("ROC-AUC", f"{metrics.get('ROC-AUC', 0):.3f}")
            st.metric("F1 Score", f"{metrics.get('F1 Score', 0):.3f}")
            st.metric("Accuracy", f"{metrics.get('Accuracy', 0):.3f}")
        
        if 'optimal_threshold' in model_data:
            st.metric("Optimal Threshold", f"{model_data['optimal_threshold']:.2f}")
        
        st.markdown("---")
        st.info("💡 **Tip:** Fill in the student information to get attendance prediction.")
    
    # Main content - Two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Student Information")
        
        # Demographics
        with st.expander("🎯 Demographics", expanded=True):
            age = st.number_input("Age", min_value=16, max_value=40, value=21, step=1)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            year = st.selectbox("Year of Study", [1, 2, 3, 4, 5])
            school = st.selectbox("School", [
                "SCIS", "SOE", "SOSS", "SOL", "LKCSB", "SOA", "YPHSL"
            ])
        
        # Engagement History
        with st.expander("📈 Past Engagement", expanded=True):
            past_attend = st.slider("Past Attendance Count (6 months)", 0.0, 10.0, 2.0, 0.5)
            no_show = st.slider("No-Show Count (6 months)", 0.0, 10.0, 1.0, 0.5)
        
        # Preferences
        with st.expander("⚙️ Preferences", expanded=True):
            next_program = st.selectbox("Preferred Next Program", [
                "Mental Health Workshop",
                "Fitness Class",
                "Career Counseling",
                "Stress Management",
                "Social Events"
            ])
            preferred_format = st.selectbox("Preferred Format", [
                "In-person",
                "Online",
                "Hybrid"
            ])
            time_windows = st.multiselect("Preferred Time Windows", [
                "Weekday daytime",
                "Weekday evening",
                "Weekend daytime",
                "Weekend evening"
            ])
    
    with col2:
        st.subheader("🚧 Barriers & Relevance")
        
        # Barriers
        with st.expander("🚧 Barriers (0-10 scale)", expanded=True):
            barrier_time = st.slider("Time Constraint", 0, 10, 5)
            barrier_cost = st.slider("Cost Barrier", 0, 10, 3)
            barrier_motivation = st.slider("Motivation/Willpower", 0, 10, 4)
            barrier_access = st.slider("Convenience/Access", 0, 10, 3)
        
        # Relevance scores
        with st.expander("⭐ Initiative Relevance (1-10 scale)", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                rel_mhw = st.slider("Mental Health Week", 1, 10, 7)
                rel_resilience = st.slider("Resilience Framework", 1, 10, 6)
                rel_exam = st.slider("Exam Angels", 1, 10, 8)
                rel_scs = st.slider("Student Care Services", 1, 10, 7)
                rel_cosy = st.slider("Cosy Haven", 1, 10, 6)
            with col_b:
                rel_voices = st.slider("Voices Roadshows", 1, 10, 5)
                rel_peer = st.slider("Peer Helpers", 1, 10, 6)
                rel_career = st.slider("Career Compass", 1, 10, 7)
                rel_cares = st.slider("CARES Corner", 1, 10, 6)
    
    # Predict button
    st.markdown("---")
    if st.button("🔮 Predict Attendance Probability", type="primary", use_container_width=True):
        # Prepare input data
        time_windows_str = ", ".join(time_windows) if time_windows else ""
        
        input_data = pd.DataFrame([{
            'age': age,
            'gender': gender,
            'year': year,
            'school': school,
            'past_attend_count': past_attend,
            'no_show_count': no_show,
            'next_program': next_program,
            'preferred_format': preferred_format,
            'preferred_time_windows': time_windows_str,
            'barrier_time': barrier_time,
            'barrier_cost': barrier_cost,
            'barrier_motivation': barrier_motivation,
            'barrier_access': barrier_access,
            'relevance_MHW': rel_mhw,
            'relevance_Resilience': rel_resilience,
            'relevance_ExamAngels': rel_exam,
            'relevance_SCS': rel_scs,
            'relevance_CosyHaven': rel_cosy,
            'relevance_Voices': rel_voices,
            'relevance_PeerHelpers': rel_peer,
            'relevance_CareerCompass': rel_career,
            'relevance_CARES': rel_cares
        }])
        
        try:
            # Preprocess and predict
            X_prep = preprocessor.transform(input_data)
            prob = model.predict_proba(X_prep)[0, 1]
            prediction = model.predict(X_prep)[0]
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            # Determine styling based on probability
            if prob >= 0.7:
                box_class = "high-prob"
                emoji = "✅"
                status = "HIGH"
                recommendation = "**Priority invite** - This student is very likely to attend. Send personalized invitations and reminders."
            elif prob >= 0.35:
                box_class = "medium-prob"
                emoji = "⚠️"
                status = "MEDIUM"
                recommendation = "**Standard outreach** - Consider incentives or personalized messaging to increase attendance."
            else:
                box_class = "low-prob"
                emoji = "❌"
                status = "LOW"
                recommendation = "**Light touch** - May need different engagement strategies or timing to increase interest."
            
            # Create columns for results
            res_col1, res_col2, res_col3 = st.columns([1, 1, 2])
            
            with res_col1:
                st.metric("Attendance Probability", f"{prob*100:.1f}%", 
                         delta=f"{status} likelihood")
            
            with res_col2:
                st.metric("Prediction", 
                         "WILL ATTEND ✓" if prediction == 1 else "WON'T ATTEND ✗")
            
            with res_col3:
                threshold = model_data.get('optimal_threshold', 0.5)
                st.metric("Threshold Used", f"{threshold:.2f}", 
                         help="Optimal threshold from model training")
            
            # Detailed box
            st.markdown(f"""
                <div class="prediction-box {box_class}">
                    <h3>{emoji} {status} Probability Student</h3>
                    <p><strong>Recommendation:</strong> {recommendation}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Key factors
            st.markdown("### 🔑 Key Insights")
            insights_col1, insights_col2 = st.columns(2)
            
            with insights_col1:
                st.markdown("**Positive Factors:**")
                if past_attend > 2:
                    st.markdown("- ✅ Good past attendance history")
                if barrier_motivation < 5:
                    st.markdown("- ✅ Low motivation barrier")
                if max([rel_mhw, rel_resilience, rel_exam, rel_scs, rel_cosy, 
                       rel_voices, rel_peer, rel_career, rel_cares]) >= 8:
                    st.markdown("- ✅ High relevance for some initiatives")
            
            with insights_col2:
                st.markdown("**Areas of Concern:**")
                if no_show > 2:
                    st.markdown("- ⚠️ Multiple previous no-shows")
                if barrier_time > 7:
                    st.markdown("- ⚠️ High time constraint")
                if barrier_cost > 7:
                    st.markdown("- ⚠️ Cost is a significant barrier")
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.info("Please ensure all fields are filled correctly.")

else:
    st.warning("⚠️ Please train the model first by running the training script.")
    st.code("""
    # To train the model, run your training script:
    python train_model.py
    
    # This will create 'attendance_model.pkl' file
    """, language="bash")
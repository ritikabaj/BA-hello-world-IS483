import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
import os
from pathlib import Path


# Suppress warnings
warnings.filterwarnings("ignore")


# Page configuration
st.set_page_config(
    page_title="SMU Wellness Attendance Predictor",
    page_icon="🎓",
    layout="wide"
)


# Custom CSS
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


# Helper functions for data cleaning
def range_to_float(x):
    if isinstance(x, str) and '-' in x:
        a, b = x.split('-')
        return (float(a) + float(b)) / 2
    try:
        return float(x)
    except:
        return np.nan


def year_to_int(x):
    if isinstance(x, str):
        import re
        m = re.search(r'\d+', x)
        if m:
            return int(m.group())
        return np.nan
    return x




@st.cache_resource
def load_model_components():
    """Load trained model and preprocessor, or indicate training is needed"""
    try:
        # Define the model file path
        model_path = 'test/attendance_model.pkl'  # Ensure this path is correct for your deployment

        # Check if the file exists at the expected path
        if not os.path.exists(model_path):
            return None, f"Model file not found at {model_path}. Please ensure the model is uploaded and accessible."

        # Load the model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # If loading is successful, return model data
        return model_data, None
    except FileNotFoundError:
        return None, f"Model file not found at {model_path}. Please run training first and make sure the file is in the correct location."
    except Exception as e:
        return None, f"Error loading model: {str(e)}"



# Load or check for Excel data file
@st.cache_data
def load_survey_data():
    """Load the survey data Excel file"""
    file_path = 'SMU_Survey_Final.xlsx'
    try:
        df = pd.read_excel(file_path)
        return df, None
    except FileNotFoundError:
        return None, f"Survey data file '{file_path}' not found."
    except Exception as e:
        return None, f"Error loading survey data: {str(e)}"


# Initialize session state for checkbox selections
if 'selected_students' not in st.session_state:
    st.session_state.selected_students = set()


# Initialize
model_data, model_error = load_model_components()


# Sidebar - Status and Info
with st.sidebar:
    st.header("📊 System Status")
    
    if model_data is not None:
        st.success("✅ Model Loaded")
        
        
    else:
        st.error("❌ Model Not Loaded")
        st.warning(model_error)
        st.info("Run your training script (test.py) first to generate the model.")
    
    st.markdown("---")
    st.subheader("📝 How to Use")
    st.markdown("""
    1. **Batch Tab**: Check boxes to select students
    2. **Single Tab**: Fill in individual student info
    3. Click **Predict** button
    4. Review recommendations
    """)


# Main application
if model_data is not None:
    model = model_data['model']
    preprocessor = model_data['preprocessor']
    features = model_data.get('features', [])
    
    # Create tabs for different functions - BATCH FIRST
    tab1, tab2 = st.tabs(["📊 Dataset Predictions", "🔮 New Prediction"])
    
    with tab1:
        st.subheader("📊 Select Students for Dataset Predictions")
        
        # Load survey data
        survey_data, data_error = load_survey_data()
        
        if survey_data is not None:
            st.success(f"✅ Loaded {len(survey_data)} survey responses")
            
            # Column mapping from test.py
            KEY_COLUMNS = {
                'age': 'Age',
                'gender': 'Gender', 
                'year': 'YearOfStudy',
                'school': 'School',
                'intent_90d': 'Q27_27InTheNext90DaysHowLikelyAreYouToUseAnySmuWellnessServiceCounsellingWorkshopsEvents',
                'past_attend_count': 'Q28_28InThePast6MonthsHowManyWellnessEventsDidYouAttend',
                'no_show_count': 'Q29_29InThePast6MonthsHowManyWellnessEventsDidYouSignUpButNotAttendNoShow',
                'next_program': 'Q30_30IfYouCouldJoinExactlyOneWellnessProgramNextWhichTypeOfEventWouldYouPick',
                'preferred_format': 'Q31_31WhichFormatDoYouPreferMost',
                'preferred_time_windows': 'Q32_32WhichTimesWorkBestForYouSelectUpTo2',
                'barrier_time': 'Q33_lackOfTime',
                'barrier_cost': 'Q33_cost',
                'barrier_motivation': 'Q33_lackOfMotivationWillpower',
                'barrier_access': 'Q33_convenience',
            }
            
            RELEVANCE_COLUMNS = {
                'relevance_MHW': 'Q34_mentalHealthWeek',
                'relevance_Resilience': 'Q34_resilienceFramework',
                'relevance_ExamAngels': 'Q34_examAngels',
                'relevance_SCS': 'Q34_studentCareServices',
                'relevance_CosyHaven': 'Q34_cosyHaven',
                'relevance_Voices': 'Q34_voicesRoadshows',
                'relevance_PeerHelpers': 'Q34_peerHelpersRoadshows',
                'relevance_CareerCompass': 'Q34_careerCompass',
                'relevance_CARES': 'Q34_caresCorner'
            }
            
            # Create a preview dataframe with student info
            preview_cols = ['Age', 'Gender', 'YearOfStudy', 'School']
            if all(col in survey_data.columns for col in preview_cols):
                preview_df = survey_data[preview_cols].copy()
                preview_df.insert(0, 'Student_ID', range(1, len(preview_df) + 1))
                preview_df['Original_Index'] = survey_data.index
                preview_df.columns = ['ID', 'Age', 'Gender', 'Year', 'School', 'Original_Index']
            else:
                preview_df = pd.DataFrame({
                    'ID': range(1, len(survey_data) + 1),
                    'Original_Index': survey_data.index
                })
            
            # Filter options
            st.markdown("### 🔍 Filter Students")
            filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
            
            with filter_col1:
                if 'School' in preview_df.columns:
                    schools = ['All'] + sorted(preview_df['School'].dropna().unique().tolist())
                    selected_school = st.selectbox("Filter by School", schools, key="filter_school")
                else:
                    selected_school = 'All'
            
            with filter_col2:
                if 'Year' in preview_df.columns:
                    years = ['All'] + sorted([str(y) for y in preview_df['Year'].dropna().unique() if pd.notna(y)])
                    selected_year = st.selectbox("Filter by Year", years, key="filter_year")
                else:
                    selected_year = 'All'
            
            with filter_col3:
                if 'Gender' in preview_df.columns:
                    genders = ['All'] + sorted(preview_df['Gender'].dropna().unique().tolist())
                    selected_gender = st.selectbox("Filter by Gender", genders, key="filter_gender")
                else:
                    selected_gender = 'All'
            
            with filter_col4:
                num_display = st.number_input("Students to display", min_value=5, max_value=100, value=20, step=5)
            
            # Apply filters
            filtered_df = preview_df.copy()
            if selected_school != 'All' and 'School' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['School'] == selected_school]
            if selected_year != 'All' and 'Year' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Year'].astype(str) == selected_year]
            if selected_gender != 'All' and 'Gender' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Gender'] == selected_gender]
            
            # Limit display
            filtered_df = filtered_df.head(num_display)
            
            st.info(f"📋 Showing {len(filtered_df)} students (filtered from {len(preview_df)} total)")
            
            # Selection controls
            sel_col1, sel_col2, sel_col3 = st.columns([1, 1, 2])
            with sel_col1:
                if st.button("✅ Select All", use_container_width=True):
                    for idx in filtered_df.index:
                        st.session_state.selected_students.add(idx)
                    st.rerun()
            
            with sel_col2:
                if st.button("❌ Clear All", use_container_width=True):
                    st.session_state.selected_students.clear()
                    st.rerun()
            
            with sel_col3:
                st.metric("Selected Students", len(st.session_state.selected_students))
            
            # Display students with checkboxes
            st.markdown("### 👥 Select Students")
            st.caption("Check the boxes to select students for prediction")
            
            # Create scrollable container
            with st.container():
                # Header row
                header_cols = st.columns([0.5, 1, 1, 1.5, 1, 2])
                header_cols[0].markdown("**Select**")
                header_cols[1].markdown("**ID**")
                header_cols[2].markdown("**Age**")
                header_cols[3].markdown("**Gender**")
                header_cols[4].markdown("**Year**")
                header_cols[5].markdown("**School**")
                
                st.markdown("---")
                
                # Student rows
                for idx, row in filtered_df.iterrows():
                    cols = st.columns([0.5, 1, 1, 1.5, 1, 2])
                    
                    # Checkbox in first column
                    with cols[0]:
                        checked = st.checkbox(
                            "Select",
                            value=idx in st.session_state.selected_students,
                            key=f"check_{idx}",
                            label_visibility="collapsed"
                        )
                        
                        if checked and idx not in st.session_state.selected_students:
                            st.session_state.selected_students.add(idx)
                        elif not checked and idx in st.session_state.selected_students:
                            st.session_state.selected_students.remove(idx)
                    
                    # Student data in other columns
                    cols[1].write(f"{row['ID']}")
                    cols[2].write(f"{row['Age']}")
                    cols[3].write(f"{row['Gender']}")
                    cols[4].write(f"{row['Year']}")
                    cols[5].write(f"{row['School']}")
            
            # Show selected students and predict button
            if len(st.session_state.selected_students) > 0:
                st.markdown("---")
                st.success(f"✅ {len(st.session_state.selected_students)} student(s) selected")
                
                # Show selected students summary
                with st.expander("📋 View Selected Students", expanded=False):
                    selected_indices = list(st.session_state.selected_students)
                    selected_display = preview_df.loc[selected_indices][['ID', 'Age', 'Gender', 'Year', 'School']]
                    st.dataframe(selected_display, use_container_width=True, hide_index=True)
                
                # Predict button
                if st.button("🎯 Generate Predictions for Selected Students", type="primary", use_container_width=True, key="predict_batch"):
                    try:
                        with st.spinner("🔄 Generating predictions..."):
                            # Get selected students from survey_data
                            selected_indices = list(st.session_state.selected_students)
                            original_indices = preview_df.loc[selected_indices]['Original_Index'].values
                            sample_students = survey_data.loc[original_indices]
                            
                            use_cols = {**KEY_COLUMNS, **RELEVANCE_COLUMNS}
                            sample_data = sample_students[list(use_cols.values())].rename(
                                columns={v:k for k,v in use_cols.items()}
                            )
                            
                            # Clean data
                            for col in ['past_attend_count', 'no_show_count']:
                                if col in sample_data.columns:
                                    sample_data[col] = sample_data[col].apply(range_to_float)
                            
                            if 'year' in sample_data.columns:
                                sample_data['year'] = sample_data['year'].apply(year_to_int)
                            
                            if 'age' in sample_data.columns:
                                sample_data['age'] = pd.to_numeric(sample_data['age'], errors='coerce')
                            
                            # Predict
                            X_sample = sample_data[features]
                            X_sample_prep = preprocessor.transform(X_sample)
                            probs = model.predict_proba(X_sample_prep)[:, 1]
                            preds = model.predict(X_sample_prep)
                            
                            # Create results DataFrame
                            selected_display_data = preview_df.loc[selected_indices]
                            results = pd.DataFrame({
                                'Student_ID': selected_display_data['ID'].values,
                                'School': sample_data['school'].values,
                                'Age': sample_data['age'].values,
                                'Year': sample_data['year'].values,
                                'Gender': sample_data['gender'].values,
                                'Probability': probs,
                                'Probability_Pct': [f"{p*100:.1f}%" for p in probs],
                                'Prediction': ['✓ WILL ATTEND' if p == 1 else '✗ WON\'T ATTEND' for p in preds],
                                'Risk_Level': ['HIGH' if p >= 0.7 else 'MEDIUM' if p >= 0.35 else 'LOW' for p in probs],
                                'Past_Attend': sample_data['past_attend_count'].values,
                                'No_Shows': sample_data['no_show_count'].values
                            })
                            
                            # Sort by probability descending
                            results = results.sort_values('Probability', ascending=False)
                            
                            st.markdown("---")
                            st.markdown("### 📊 Prediction Results")
                            
                            # Display results with color coding
                            display_results = results[['Student_ID', 'School', 'Age', 'Year', 'Gender', 
                                                       'Probability_Pct', 'Prediction', 'Risk_Level', 
                                                       'Past_Attend', 'No_Shows']].copy()
                            
                            st.dataframe(
                                display_results,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Student_ID": st.column_config.TextColumn("Student ID", width="small"),
                                    "School": st.column_config.TextColumn("School", width="small"),
                                    "Age": st.column_config.NumberColumn("Age", width="small"),
                                    "Year": st.column_config.NumberColumn("Year", width="small"),
                                    "Gender": st.column_config.TextColumn("Gender", width="small"),
                                    "Probability_Pct": st.column_config.TextColumn("Probability", width="medium"),
                                    "Prediction": st.column_config.TextColumn("Prediction", width="medium"),
                                    "Risk_Level": st.column_config.TextColumn("Priority", width="small"),
                                    "Past_Attend": st.column_config.NumberColumn("Past Attendance", width="small"),
                                    "No_Shows": st.column_config.NumberColumn("No-Shows", width="small"),
                                }
                            )
                            
                            # Summary statistics
                            st.markdown("### 📈 Summary Statistics")
                            sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
                            
                            with sum_col1:
                                will_attend = (preds == 1).sum()
                                st.metric("Will Attend", f"{will_attend}/{len(preds)}", 
                                         delta=f"{will_attend/len(preds)*100:.1f}%")
                            
                            with sum_col2:
                                avg_prob = probs.mean()
                                st.metric("Average Probability", f"{avg_prob*100:.1f}%")
                            
                            with sum_col3:
                                high_priority = (probs >= 0.7).sum()
                                st.metric("High Priority", f"{high_priority}/{len(probs)}", 
                                         delta=f"{high_priority/len(probs)*100:.1f}%")
                            
                            with sum_col4:
                                low_priority = (probs < 0.35).sum()
                                st.metric("Low Priority", f"{low_priority}/{len(probs)}", 
                                         delta=f"{low_priority/len(probs)*100:.1f}%")
                            
                            # Segmentation recommendations
                            st.markdown("### 💡 Recommendations by Priority")
                            
                            rec_col1, rec_col2, rec_col3 = st.columns(3)
                            
                            with rec_col1:
                                high_students = results[results['Risk_Level'] == 'HIGH']
                                st.markdown(f"""
                                    <div class="prediction-box high-prob">
                                        <h4>✅ HIGH Priority ({len(high_students)} students)</h4>
                                        <p><strong>Strategy:</strong> Priority invitations with personalized messages. 
                                        These students are highly likely to attend.</p>
                                        <p><strong>Action:</strong> Send early invitations and reminders.</p>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            with rec_col2:
                                medium_students = results[results['Risk_Level'] == 'MEDIUM']
                                st.markdown(f"""
                                    <div class="prediction-box medium-prob">
                                        <h4>⚠️ MEDIUM Priority ({len(medium_students)} students)</h4>
                                        <p><strong>Strategy:</strong> Standard outreach with incentives. 
                                        Consider addressing barriers like time or motivation.</p>
                                        <p><strong>Action:</strong> Highlight program benefits and flexible timing.</p>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            with rec_col3:
                                low_students = results[results['Risk_Level'] == 'LOW']
                                st.markdown(f"""
                                    <div class="prediction-box low-prob">
                                        <h4>❌ LOW Priority ({len(low_students)} students)</h4>
                                        <p><strong>Strategy:</strong> Light touch or alternative engagement. 
                                        May need different program types or timing.</p>
                                        <p><strong>Action:</strong> Explore barriers and preferences through surveys.</p>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            # Download button
                            st.markdown("---")
                            csv = results.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name=f"attendance_predictions_{len(results)}_students.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        
                    except Exception as e:
                        st.error(f"Error during batch prediction: {str(e)}")
                        st.exception(e)
                        st.info("Please check your data and model compatibility.")
            else:
                st.info("👆 Check the boxes above to select students for prediction")
        
        else:
            st.error(data_error)
            st.info("Make sure 'SMU_Survey_Final.xlsx' is in the same directory as this app.")
    
    with tab2:
        # Single student prediction interface
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
                st.info("Enter average values or use ranges (e.g., 2-3 becomes 2.5)")
                past_attend = st.number_input("Past Attendance Count (6 months)", 
                                             min_value=0.0, max_value=10.0, value=2.0, step=0.5)
                no_show = st.number_input("No-Show Count (6 months)", 
                                         min_value=0.0, max_value=10.0, value=1.0, step=0.5)
            
            # Preferences
            with st.expander("⚙️ Preferences", expanded=True):
                next_program = st.selectbox("Preferred Next Program", [
                    "Mental Health Workshop",
                    "Fitness Class",
                    "Career Counseling",
                    "Stress Management",
                    "Social Events",
                    "Peer Support",
                    "Academic Support"
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
                ], default=["Weekday evening"])
        
        with col2:
            st.subheader("🚧 Barriers & Relevance")
            
            # Barriers
            with st.expander("🚧 Barriers (0-10 scale)", expanded=True):
                st.caption("Higher = Bigger barrier")
                barrier_time = st.slider("Time Constraint", 0, 10, 5)
                barrier_cost = st.slider("Cost Barrier", 0, 10, 3)
                barrier_motivation = st.slider("Motivation/Willpower", 0, 10, 4)
                barrier_access = st.slider("Convenience/Access", 0, 10, 3)
            
            # Relevance scores
            with st.expander("⭐ Initiative Relevance (1-10 scale)", expanded=True):
                st.caption("How relevant is each initiative to this student?")
                col_a, col_b = st.columns(2)
                with col_a:
                    rel_mhw = st.slider("Mental Health Week", 1, 10, 7, key='mhw')
                    rel_resilience = st.slider("Resilience Framework", 1, 10, 6, key='res')
                    rel_exam = st.slider("Exam Angels", 1, 10, 8, key='exam')
                    rel_scs = st.slider("Student Care Services", 1, 10, 7, key='scs')
                    rel_cosy = st.slider("Cosy Haven", 1, 10, 6, key='cosy')
                with col_b:
                    rel_voices = st.slider("Voices Roadshows", 1, 10, 5, key='voices')
                    rel_peer = st.slider("Peer Helpers", 1, 10, 6, key='peer')
                    rel_career = st.slider("Career Compass", 1, 10, 7, key='career')
                    rel_cares = st.slider("CARES Corner", 1, 10, 6, key='cares')
        
        # Predict button
        st.markdown("---")
        if st.button("🔮 Predict Attendance Probability", type="primary", use_container_width=True, key="predict_single"):
            # Prepare input data matching your feature structure
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
                threshold = model_data.get('optimal_threshold', 0.5)
                
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
                    prediction_text = "WILL ATTEND ✓" if prediction == 1 else "WON'T ATTEND ✗"
                    st.metric("Prediction", prediction_text)
                
                with res_col3:
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
                    positive_factors = []
                    if past_attend > 2:
                        positive_factors.append("✅ Good past attendance history")
                    if barrier_motivation < 5:
                        positive_factors.append("✅ Low motivation barrier")
                    if barrier_time < 5:
                        positive_factors.append("✅ Time is not a major constraint")
                    max_relevance = max([rel_mhw, rel_resilience, rel_exam, rel_scs, 
                                        rel_cosy, rel_voices, rel_peer, rel_career, rel_cares])
                    if max_relevance >= 8:
                        positive_factors.append("✅ High relevance for some initiatives")
                    
                    if positive_factors:
                        for factor in positive_factors:
                            st.markdown(f"- {factor}")
                    else:
                        st.markdown("- No strong positive factors identified")
                
                with insights_col2:
                    st.markdown("**Areas of Concern:**")
                    concerns = []
                    if no_show > 2:
                        concerns.append("⚠️ Multiple previous no-shows")
                    if barrier_time > 7:
                        concerns.append("⚠️ High time constraint")
                    if barrier_cost > 7:
                        concerns.append("⚠️ Cost is a significant barrier")
                    if barrier_motivation > 7:
                        concerns.append("⚠️ Low motivation/willpower")
                    
                    if concerns:
                        for concern in concerns:
                            st.markdown(f"- {concern}")
                    else:
                        st.markdown("- No major concerns identified")
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.exception(e)
                st.info("Please ensure all fields are filled correctly and the model is properly trained.")


else:
    # Model not loaded - show instructions
    st.error("🚫 Model Not Available")
    st.markdown("""
    ### To use this predictor, you need to train the model first:
    
    1. Make sure you have these files:
       - `test.py` (your training script)
       - `SMU_Survey_Final.xlsx` (survey data)
    
    2. Run the training script:
       ```
       python test.py
       ```
    
    3. This will create `attendance_model.pkl`
    
    4. Refresh this page or restart the app:
       ```
       streamlit run app.py
       ```
    """)
    
    st.info("💡 The training process includes data cleaning, debiasing, and XGBoost model training. It may take 2-5 minutes.")

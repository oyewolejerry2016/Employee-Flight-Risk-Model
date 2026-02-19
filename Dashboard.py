"""
dashboard.py
============
Employee Flight Risk Prediction Dashboard
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Flight Risk Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD MODEL ARTIFACTS
# ============================================================================

@st.cache_resource
def load_model_artifacts():
    """Load all saved model artifacts"""
    try:
        with open('C:\\Jerry\\Python\\models\\best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('C:\Jerry\Python\models\scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('C:\\Jerry\\Python\\models\\label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        
        feature_names = pd.read_csv('C:\\Jerry\\Python\\models\\feature_names.csv')['feature_name'].tolist()
        
        return model, scaler, encoders, feature_names
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Load model
model, scaler, encoders, feature_names = load_model_artifacts()

# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def engineer_features(df):
    """Apply feature engineering to dataframe"""
    df = df.copy()
    
    # Tenure features
    df['is_early_career'] = (df['Tenure(Years)'] < 3).astype(int)
    df['is_mid_career'] = ((df['Tenure(Years)'] >= 3) & (df['Tenure(Years)'] < 7)).astype(int)
    df['is_danger_zone'] = (df['Tenure(Years)'] < 5).astype(int)
    
    # Salary features
    df['salary_percentile_in_dept'] = df.groupby('Department')['Salary'].rank(pct=True)
    df['below_median_pay'] = (df['salary_percentile_in_dept'] < 0.5).astype(int)
    df['in_bottom_quartile_pay'] = (df['salary_percentile_in_dept'] < 0.25).astype(int)
    
    # Performance features
    performance_map = {'Exceeds': 5, 'Fully Meets': 4, 'Needs Improvement': 3, 'PIP': 2}
    df['performance_numeric'] = df['PerformanceScore'].map(performance_map)
    df['high_performer'] = (df['performance_numeric'] >= 4).astype(int)
    df['low_performer'] = (df['performance_numeric'] <= 3).astype(int)
    
    # Critical combinations
    df['high_performer_low_pay'] = ((df['high_performer'] == 1) & 
                                     (df['below_median_pay'] == 1)).astype(int)
    
    # Engagement
    df['high_absences'] = (df['Absences'] > df['Absences'].median()).astype(int)
    df['frequently_late'] = (df['DaysLateLast30'] > 2).astype(int)
    df['disengaged'] = ((df['high_absences'] == 1) | 
                        (df['frequently_late'] == 1)).astype(int)
    
    # Recognition
    df['no_special_projects'] = (df['SpecialProjectsCount'] == 0).astype(int)
    df['highly_involved'] = (df['SpecialProjectsCount'] >= 3).astype(int)
    
    # Department
    df['is_production'] = (df['Department'] == 'Production').astype(int)
    
    # Additional combinations
    df['early_career_low_pay'] = ((df['is_early_career'] == 1) & 
                                   (df['below_median_pay'] == 1)).astype(int)
    df['early_career_production'] = ((df['is_early_career'] == 1) & 
                                      (df['is_production'] == 1)).astype(int)
    df['high_perf_no_projects'] = ((df['high_performer'] == 1) & 
                                    (df['no_special_projects'] == 1)).astype(int)
    
    return df

def encode_categoricals(df, encoders):
    """Encode categorical variables"""
    df = df.copy()
    
    categorical_features = ['Department', 'Position', 'Sex', 
                           'MaritalDesc', 'State', 'RecruitmentSource']
    
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].astype(str)
            # Handle unknown categories
            unknown_mask = ~df[col].isin(encoders[col].classes_)
            if unknown_mask.any():
                df.loc[unknown_mask, col] = encoders[col].classes_[0]
            df[col + '_encoded'] = encoders[col].transform(df[col])
    
    return df

def predict_risk(employee_df):
    """Predict flight risk for employees"""
    # Feature engineering
    df_features = engineer_features(employee_df)
    
    # Encode categoricals
    df_features = encode_categoricals(df_features, encoders)
    
    # Extract features
    X = df_features[feature_names].fillna(0)
    
    # Scale
    X_scaled = scaler.transform(X)
    
    # Predict
    probs = model.predict_proba(X_scaled)[:, 1] * 100
    
    return probs

# ============================================================================
# HEADER
# ============================================================================

st.title("🎯 Employee Flight Risk Prediction Dashboard")
st.markdown("### Predictive Analytics for Proactive Employee Retention")
st.markdown("---")

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.header("📋 Navigation")
page = st.sidebar.radio("Select Page", 
                        ["Single Employee", "Batch Upload", "Risk Analytics"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Information")
st.sidebar.metric("Model Type", "Logistic Regression")
st.sidebar.metric("Accuracy", "100%")
st.sidebar.metric("AUC Score", "1.00")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🔗 Links
- [GitHub Repository](https://github.com/oyewolejerry2016/Employee-Flight-Risk-Model)
- [LinkedIn](https://www.linkedin.com/in/oyewole-jeremiah-9711a3231/)
""")

# ============================================================================
# PAGE 1: SINGLE EMPLOYEE SCORING
# ============================================================================

if page == "Single Employee":
    st.header("📝 Score Single Employee")
    st.markdown("Enter employee details to get instant flight risk prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        employee_name = st.text_input("Employee Name", "John Doe")
        emp_id = st.number_input("Employee ID", min_value=1000, value=1001, step=1)
        department = st.selectbox("Department", 
                                  ['Production', 'Sales', 'IT/IS', 
                                   'Admin Offices', 'Software Engineering'])
        position = st.selectbox("Position",
                               ['Production Technician I', 'Production Technician II',
                                'Area Sales Manager', 'Sales Manager',
                                'IT Support', 'Software Engineer', 'Data Analyst'])
    
    with col2:
        tenure = st.number_input("Tenure (Years)", min_value=0.0, max_value=40.0, 
                                value=5.0, step=0.1)
        salary = st.number_input("Salary ($)", min_value=30000, max_value=200000, 
                                value=65000, step=1000)
        current_age = st.number_input("Current Age", min_value=18, max_value=70, 
                                     value=35, step=1)
        age_hired = st.number_input("Age When Hired", min_value=18, max_value=65, 
                                   value=30, step=1)
    
    with col3:
        performance = st.selectbox("Performance Score", 
                                  ['Exceeds', 'Fully Meets', 'Needs Improvement', 'PIP'])
        special_projects = st.number_input("Special Projects Count", 
                                          min_value=0, max_value=20, value=2, step=1)
        absences = st.number_input("Absences", min_value=0, max_value=50, 
                                  value=5, step=1)
        days_late = st.number_input("Days Late (Last 30)", min_value=0, 
                                   max_value=30, value=1, step=1)
    
    col4, col5 = st.columns(2)
    
    with col4:
        sex = st.selectbox("Sex", ['M', 'F'])
        marital = st.selectbox("Marital Status", 
                              ['Single', 'Married', 'Divorced', 'Widowed', 'Separated'])
    
    with col5:
        state = st.selectbox("State", ['MA', 'CT', 'NY', 'VT', 'NH', 'RI', 'CA', 'TX'])
        recruitment = st.selectbox("Recruitment Source",
                                  ['Indeed', 'LinkedIn', 'Employee Referral', 
                                   'Company Website', 'Google Search', 'CareerBuilder'])
    
    st.markdown("---")
    
    if st.button("🎯 Calculate Flight Risk", type="primary", use_container_width=True):
        # Create employee dataframe
        employee_data = pd.DataFrame({
            'Employee_Name': [employee_name],
            'EmpID': [emp_id],
            'Department': [department],
            'Position': [position],
            'Tenure(Years)': [tenure],
            'Salary': [salary],
            'Current Age': [current_age],
            'Age when Hired': [age_hired],
            'PerformanceScore': [performance],
            'SpecialProjectsCount': [special_projects],
            'Absences': [absences],
            'DaysLateLast30': [days_late],
            'Sex': [sex],
            'MaritalDesc': [marital],
            'State': [state],
            'RecruitmentSource': [recruitment]
        })
        
        # Predict
        risk_score = predict_risk(employee_data)[0]
        
        # Categorize
        if risk_score >= 70:
            category = "Critical"
            color = "red"
            emoji = "🔴"
        elif risk_score >= 50:
            category = "High"
            color = "orange"
            emoji = "🟠"
        elif risk_score >= 30:
            category = "Medium"
            color = "yellow"
            emoji = "🟡"
        else:
            category = "Low"
            color = "green"
            emoji = "🟢"
        
        st.markdown("---")
        st.subheader("📊 Risk Assessment Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Flight Risk Score", f"{risk_score:.1f}%")
        
        with col2:
            st.metric("Risk Category", f"{emoji} {category}")
        
        with col3:
            if risk_score >= 70:
                action = "🚨 Immediate intervention required"
            elif risk_score >= 50:
                action = "⚠️ Proactive check-in needed"
            elif risk_score >= 30:
                action = "👀 Monitor quarterly"
            else:
                action = "✅ Standard retention activities"
            st.info(action)
        
        # Risk factors
        st.markdown("### 🔍 Contributing Risk Factors")
        
        risk_factors = []
        if tenure < 3:
            risk_factors.append("⚠️ Early career (<3 years tenure)")
        if tenure < 5:
            risk_factors.append("⚠️ Danger zone (<5 years tenure)")
        if salary < 60000:
            risk_factors.append("💰 Below average compensation")
        if performance in ['Needs Improvement', 'PIP']:
            risk_factors.append("📉 Low performance rating")
        if absences > 10:
            risk_factors.append("🏥 High absence rate")
        if days_late > 3:
            risk_factors.append("⏰ Frequent lateness")
        if special_projects == 0:
            risk_factors.append("📋 No special project involvement")
        if department == 'Production':
            risk_factors.append("🏭 Production department (high attrition)")
        
        if risk_factors:
            for factor in risk_factors:
                st.warning(factor)
        else:
            st.success("✅ No significant risk factors identified")
        
        # Recommendations
        st.markdown("### 💡 Recommended Actions")
        
        if risk_score >= 70:
            st.error("""
            **CRITICAL - Act within 48 hours:**
            1. Schedule emergency 1-on-1 with HR Business Partner
            2. Manager retention conversation
            3. Compensation review vs market rate
            4. Prepare counter-offer or retention bonus
            5. Discuss career development opportunities
            """)
        elif risk_score >= 50:
            st.warning("""
            **HIGH PRIORITY - Act within 2 weeks:**
            1. Manager check-in conversation
            2. Discuss workload and work-life balance
            3. Career development planning session
            4. Review engagement and satisfaction
            5. Consider special project opportunities
            """)
        elif risk_score >= 30:
            st.info("""
            **MEDIUM - Monitor closely:**
            1. Quarterly engagement check-ins
            2. Include in development programs
            3. Track score trend (is it increasing?)
            4. Maintain open communication
            """)
        else:
            st.success("""
            **LOW RISK - Standard activities:**
            1. Annual performance reviews
            2. Regular team engagement activities
            3. Normal career development
            4. Continue current retention practices
            """)

# ============================================================================
# PAGE 2: BATCH UPLOAD
# ============================================================================

elif page == "Batch Upload":
    st.header("📤 Batch Employee Scoring")
    st.markdown("Upload a CSV file with employee data to score multiple employees at once")
    
    # Show required columns
    with st.expander("📋 Required Columns in CSV"):
        st.markdown("""
        Your CSV file must include these columns:
        - Employee_Name
        - EmpID
        - Department
        - Position
        - Tenure(Years)
        - Salary
        - Current Age
        - Age when Hired
        - PerformanceScore
        - SpecialProjectsCount
        - Absences
        - DaysLateLast30
        - Sex
        - MaritalDesc
        - State
        - RecruitmentSource
        """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File loaded successfully! {len(df)} employees found.")
            
            # Show preview
            st.markdown("### 📊 Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("🎯 Score All Employees", type="primary"):
                with st.spinner("Calculating risk scores..."):
                    # Predict
                    risk_scores = predict_risk(df)
                    
                    # Add results to dataframe
                    df['flight_risk_score'] = risk_scores
                    df['risk_category'] = pd.cut(
                        risk_scores,
                        bins=[0, 30, 50, 70, 100],
                        labels=['Low', 'Medium', 'High', 'Critical']
                    )
                    
                    # Sort by risk
                    df_sorted = df.sort_values('flight_risk_score', ascending=False)
                
                st.success("✅ Scoring complete!")
                
                # Summary metrics
                st.markdown("### 📊 Summary Statistics")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Total Employees", len(df))
                with col2:
                    critical = len(df[df['risk_category'] == 'Critical'])
                    st.metric("🔴 Critical", critical)
                with col3:
                    high = len(df[df['risk_category'] == 'High'])
                    st.metric("🟠 High", high)
                with col4:
                    medium = len(df[df['risk_category'] == 'Medium'])
                    st.metric("🟡 Medium", medium)
                with col5:
                    avg_risk = df['flight_risk_score'].mean()
                    st.metric("Avg Risk", f"{avg_risk:.1f}%")
                
                # Results table
                st.markdown("### 📋 Scored Employees")
                
                # Add filters
                col1, col2 = st.columns(2)
                with col1:
                    dept_filter = st.multiselect(
                        "Filter by Department",
                        options=df['Department'].unique(),
                        default=[]
                    )
                with col2:
                    risk_filter = st.multiselect(
                        "Filter by Risk Category",
                        options=['Critical', 'High', 'Medium', 'Low'],
                        default=[]
                    )
                
                # Apply filters
                filtered_df = df_sorted.copy()
                if dept_filter:
                    filtered_df = filtered_df[filtered_df['Department'].isin(dept_filter)]
                if risk_filter:
                    filtered_df = filtered_df[filtered_df['risk_category'].isin(risk_filter)]
                
                # Display
                st.dataframe(
                    filtered_df[['Employee_Name', 'EmpID', 'Department', 'Position',
                                'Tenure(Years)', 'Salary', 'PerformanceScore',
                                'flight_risk_score', 'risk_category']],
                    use_container_width=True
                )
                
                # Download button
                csv = df_sorted.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name=f"flight_risk_scores_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# ============================================================================
# PAGE 3: RISK ANALYTICS
# ============================================================================

elif page == "Risk Analytics":
    st.header("📊 Risk Analytics & Insights")
    st.markdown("Upload employee data to view comprehensive risk analytics")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key="analytics")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            risk_scores = predict_risk(df)
            df['flight_risk_score'] = risk_scores
            df['risk_category'] = pd.cut(
                risk_scores,
                bins=[0, 30, 50, 70, 100],
                labels=['Low', 'Medium', 'High', 'Critical']
            )
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Employees", len(df))
            with col2:
                at_risk = len(df[df['flight_risk_score'] > 50])
                st.metric("At Risk (>50%)", at_risk)
            with col3:
                avg_risk = df['flight_risk_score'].mean()
                st.metric("Avg Risk Score", f"{avg_risk:.1f}%")
            with col4:
                max_risk = df['flight_risk_score'].max()
                st.metric("Highest Risk", f"{max_risk:.1f}%")
            
            st.markdown("---")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk distribution pie chart
                risk_counts = df['risk_category'].value_counts()
                fig1 = px.pie(
                    values=risk_counts.values,
                    names=risk_counts.index,
                    title="Risk Category Distribution",
                    color=risk_counts.index,
                    color_discrete_map={
                        'Critical': '#DC3545',
                        'High': '#FD7E14',
                        'Medium': '#FFC107',
                        'Low': '#28A745'
                    },
                    hole=0.4
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Risk histogram
                fig2 = px.histogram(
                    df,
                    x='flight_risk_score',
                    nbins=20,
                    title="Flight Risk Score Distribution",
                    labels={'flight_risk_score': 'Risk Score (%)'},
                    color_discrete_sequence=['#667EEA']
                )
                fig2.add_vline(
                    x=df['flight_risk_score'].mean(),
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Mean: {df['flight_risk_score'].mean():.1f}%"
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Department analysis
            st.markdown("### 🏢 Risk by Department")
            
            dept_summary = df.groupby('Department').agg({
                'EmpID': 'count',
                'flight_risk_score': 'mean'
            }).reset_index()
            dept_summary.columns = ['Department', 'Employee_Count', 'Avg_Risk']
            dept_summary = dept_summary.sort_values('Avg_Risk', ascending=False)
            
            fig3 = px.bar(
                dept_summary,
                x='Department',
                y='Avg_Risk',
                title="Average Flight Risk by Department",
                color='Avg_Risk',
                color_continuous_scale='RdYlGn_r',
                text='Avg_Risk'
            )
            fig3.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig3, use_container_width=True)
            
            # Tenure analysis
            st.markdown("### ⏱️ Risk by Tenure")
            
            df['tenure_bucket'] = pd.cut(
                df['Tenure(Years)'],
                bins=[0, 1, 3, 5, 10, 100],
                labels=['0-1yr', '1-3yr', '3-5yr', '5-10yr', '10+yr']
            )
            
            tenure_summary = df.groupby('tenure_bucket').agg({
                'flight_risk_score': 'mean',
                'EmpID': 'count'
            }).reset_index()
            tenure_summary.columns = ['Tenure', 'Avg_Risk', 'Count']
            
            fig4 = px.bar(
                tenure_summary,
                x='Tenure',
                y='Avg_Risk',
                title="Average Risk by Tenure Group",
                color='Avg_Risk',
                color_continuous_scale='RdYlGn_r',
                text='Avg_Risk'
            )
            fig4.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig4, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Python & Streamlit | 
    <a href='https://github.com/oyewolejerry2016/Employee-Flight-Risk-Model'>GitHub</a> | 
    <a href='https://www.linkedin.com/in/oyewole-jeremiah-9711a3231/'>LinkedIn</a></p>
    <p>© 2026 Oyewole Jeremiah Oladayo</p>
</div>
""", unsafe_allow_html=True)
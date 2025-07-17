import streamlit as st
import base64
import json
import io
from fpdf import FPDF
import re
import os
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textstat
from difflib import SequenceMatcher
import hashlib  # Already imported in your code

# Download NLP resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# MUST BE FIRST - Page Configuration
st.set_page_config(
    page_title="Resume Analyzer Pro",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- USER AUTHENTICATION SYSTEM ---
if "users" not in st.session_state:
    # Demo user (password: demo123)
    demo_password = "demo123"
    hashed_demo_password = hashlib.sha256(demo_password.encode()).hexdigest()
    st.session_state["users"] = {"demo@example.com": {"password": hashed_demo_password}}

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def signup(email, password):
    users = st.session_state["users"]
    if email in users:
        return False, "User already exists."
    users[email] = {"password": hash_password(password)}
    return True, "Signup successful! Please log in."

def login(email, password):
    users = st.session_state["users"]
    if email in users and users[email]["password"] == hash_password(password):
        return True, "Login successful!"
    return False, "Invalid email or password."

# --- Login/Signup UI ---
def login_signup_ui():
    st.markdown("""
    <div style="max-width: 400px; margin: 60px auto; background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                border-radius: 16px; padding: 36px 32px; border: 1px solid #334155; box-shadow: 0 4px 16px rgba(0,0,0,0.13);">
        <h2 style="color: #3b82f6; text-align: center; margin-bottom: 24px;">🔐 Resume Analyzer Login</h2>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    login_success = False

    with tab1:
        login_email = st.text_input("Email", key="login_email")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", key="login_btn", use_container_width=True):
            ok, msg = login(login_email, login_password)
            if ok:
                st.session_state["logged_in"] = True
                st.session_state["user_email"] = login_email
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    with tab2:
        signup_email = st.text_input("Email", key="signup_email")
        signup_password = st.text_input("Password", type="password", key="signup_password")
        signup_password2 = st.text_input("Confirm Password", type="password", key="signup_password2")
        if st.button("Sign Up", key="signup_btn", use_container_width=True):
            if signup_password != signup_password2:
                st.error("Passwords do not match.")
            elif len(signup_password) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                ok, msg = signup(signup_email, signup_password)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

# --- Require login before showing the app ---
if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    login_signup_ui()
    st.stop()

# Custom CSS for professional styling (rest of your CSS remains the same)
st.markdown("""
<style>
    /* --- Modern Glassmorphism Background --- */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f1f5f9;
        min-height: 100vh;
        font-family: 'Poppins', 'Segoe UI', Arial, sans-serif;
    }

    /* --- Custom Scrollbar --- */
    ::-webkit-scrollbar {
        width: 10px;
        background: #1e293b;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
        border-radius: 8px;
    }

    /* --- Responsive Flex Card Layout --- */
    .custom-flex-cards {
        display: flex;
        flex-wrap: wrap;
        gap: 32px;
        justify-content: center;
        align-items: stretch;
        margin: 0 auto;
        max-width: 1400px;
    }
    .custom-card {
        background: rgba(30, 41, 59, 0.95);
        border: 2.5px solid #6366f1;
        border-radius: 26px;
        box-shadow: 0 12px 40px 0 rgba(59, 130, 246, 0.18), 0 2px 8px #6366f155;
        min-width: 260px;
        max-width: 370px;
        flex: 1 1 340px;
        padding: 34px 28px;
        margin: 0;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        transition: box-shadow 0.3s, border 0.3s, transform 0.2s;
        overflow: hidden;
        min-height: 340px;
        word-break: break-word;
        box-sizing: border-box;
        position: relative;
        z-index: 1;
    }
    .custom-card::before {
        content: "";
        position: absolute;
        top: -40px; left: -40px;
        width: 120px; height: 120px;
        background: radial-gradient(circle, #6366f1 0%, transparent 70%);
        opacity: 0.13;
        z-index: 0;
        border-radius: 50%;
    }
    .custom-card::after {
        content: "";
        position: absolute;
        bottom: -40px; right: -40px;
        width: 120px; height: 120px;
        background: radial-gradient(circle, #3b82f6 0%, transparent 70%);
        opacity: 0.13;
        z-index: 0;
        border-radius: 50%;
    }
    .custom-card h4 {
        color: #fff;
        font-size: 1.45rem;
        margin-bottom: 18px;
        font-weight: 900;
        letter-spacing: 0.5px;
        text-shadow: 0 2px 12px #6366f1aa;
        white-space: pre-line;
        text-align: center;
        line-height: 1.2;
        word-break: break-word;
        z-index: 2;
        position: relative;
    }
    .custom-card ul {
        color: #c7d2fe;
        font-size: 1.11rem;
        padding-left: 0;
        margin: 0;
        list-style: none;
        word-break: break-word;
        display: flex;
        flex-direction: column;
        gap: 13px;
        z-index: 2;
        position: relative;
    }
    .custom-card ul li {
        margin-bottom: 0;
        line-height: 1.7;
        padding: 12px 16px;
        border-radius: 12px;
        background: linear-gradient(90deg, #334155 0%, #1e293b 100%);
        box-shadow: 0 2px 8px #33415533;
        font-size: 1.04rem;
        font-weight: 600;
        color: #e0e7ef;
        word-break: break-word;
        overflow-wrap: break-word;
        text-align: left;
        transition: background 0.2s, color 0.2s;
        border-left: 4px solid #6366f1;
    }
    .custom-card ul li:hover {
        background: linear-gradient(90deg, #6366f1 0%, #3b82f6 100%);
        color: #fff;
        border-left: 4px solid #3b82f6;
    }

    /* --- Neon Gradient Title --- */
    .main-title {
        font-weight: 900;
        font-size: 62px;
        background: linear-gradient(90deg, #3b82f6 0%, #06b6d4 50%, #a21caf 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 4px 32px #3b82f633, 0 1px 0 #fff;
        letter-spacing: 2px;
        margin-bottom: 18px;
        text-align: center;
        animation: neon-glow 2.5s infinite alternate;
    }
    @keyframes neon-glow {
        from { text-shadow: 0 0 8px #3b82f6, 0 0 16px #6366f1; }
        to   { text-shadow: 0 0 32px #06b6d4, 0 0 48px #a21caf; }
    }

    /* --- Animated Skill Badges --- */
    .skill-badge {
        background: linear-gradient(90deg, #3b82f6 0%, #06b6d4 100%);
        color: #fff;
        padding: 10px 22px;
        border-radius: 50px;
        margin: 8px 6px;
        font-size: 1rem;
        font-weight: 700;
        box-shadow: 0 2px 12px #3b82f655;
        display: inline-block;
        transition: background 0.3s, transform 0.2s;
        animation: badge-pop 1.2s cubic-bezier(.68,-0.55,.27,1.55) both;
    }
    .skill-badge:hover {
        background: linear-gradient(90deg, #a21caf 0%, #6366f1 100%);
        transform: scale(1.08) rotate(-2deg);
        box-shadow: 0 4px 24px #a21caf55;
    }
    @keyframes badge-pop {
        0% { transform: scale(0.7); opacity: 0; }
        80% { transform: scale(1.1); opacity: 1; }
        100% { transform: scale(1); }
    }

    /* --- Download Buttons --- */
    .download-btn {
        display: inline-block;
        background: linear-gradient(90deg, #3b82f6 0%, #06b6d4 100%);
        color: #fff;
        padding: 16px 38px;
        border-radius: 14px;
        font-weight: 900;
        font-size: 1.15rem;
        text-align: center;
        margin: 16px 10px;
        text-decoration: none;
        box-shadow: 0 2px 16px #3b82f655;
        transition: background 0.3s, transform 0.2s;
        border: none;
    }
    .download-btn:hover {
        background: linear-gradient(90deg, #a21caf 0%, #6366f1 100%);
        color: #fff;
        transform: scale(1.07);
        text-decoration: none;
        box-shadow: 0 4px 24px #a21caf55;
    }

    /* --- ATS Score Card --- */
    .ats-score-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 2px solid #3b82f6;
        border-radius: 22px;
        padding: 32px;
        margin: 32px 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.18);
        text-align: center;
    }
    .ats-score-value {
        font-size: 54px;
        font-weight: 900;
        background: linear-gradient(90deg, #3b82f6 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        letter-spacing: 1.5px;
    }
    .ats-score-label {
        color: #94a3b8;
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    /* --- Recommendation Box --- */
    .recommendation-box {
        background: linear-gradient(90deg, #3b82f611 0%, #06b6d411 100%);
        border-left: 6px solid #3b82f6;
        padding: 22px 28px;
        border-radius: 14px;
        margin: 18px 0;
        font-size: 1.13rem;
        color: #f1f5f9;
        font-weight: 700;
        box-shadow: 0 2px 12px #3b82f655;
    }

    /* --- Responsive for Mobile --- */
    @media (max-width: 1100px) {
        .custom-flex-cards { flex-direction: column; align-items: center; }
        .custom-card { max-width: 98vw; min-width: 220px; }
    }
    @media (max-width: 700px) {
        .main-title { font-size: 2.2rem; }
        .custom-card { padding: 18px 8px; font-size: 0.98rem; }
        .ats-score-card { padding: 16px; }
        .download-btn { padding: 10px 16px; font-size: 1rem; }
    }
</style>
""", unsafe_allow_html=True)

# Add logout button to sidebar
with st.sidebar:
    if st.session_state.get("logged_in"):
        st.markdown(f"### 👤 Welcome, {st.session_state['user_email']}")
        if st.button("Logout", key="logout_btn", use_container_width=True):
            st.session_state["logged_in"] = False
            st.session_state.pop("user_email", None)
            st.rerun()

# ---- IMPORT YOUR CORE LOGIC ----
# Assuming dummy.py exists and contains the JobMatchingSystem class
from dummy import JobMatchingSystem

# --- File Reading Libraries ---
try:
    import PyPDF2
except ImportError:
    st.error("PyPDF2 is not installed. Please run: pip install PyPDF2")
    st.stop()
try:
    import docx
except ImportError:
    st.error("python-docx is not installed. Please run: pip install python-docx")
    st.stop()

# CACHED FUNCTION TO LOAD THE JobMatchingSystem
@st.cache_resource
def load_jms_cached():
    system = JobMatchingSystem()
    return system

jms_system = load_jms_cached()

# --- Helper Functions ---
def create_skills_pie_chart(skills_data):
    if not skills_data or len(skills_data) == 0:
        return None

    # Count skill frequencies
    skill_counts = pd.Series(skills_data).value_counts().reset_index()
    skill_counts.columns = ['Skill', 'Count']

    # Take top 8 skills
    top_skills = skill_counts.head(8)

    fig = px.pie(
        top_skills,
        values='Count',
        names='Skill',
        title="Top Skills Distribution",
        color_discrete_sequence=px.colors.qualitative.Dark24,
        hover_data=['Count']
    )
    fig.update_traces(textposition='inside', textinfo='label+percent')
    
    fig.update_layout(
        height=400,
        showlegend=True,
        paper_bgcolor='#0f172a',
        plot_bgcolor='#0f172a',
        font=dict(color='#f1f5f9'),  # Light text for dark background
        title_font=dict(color='#f1f5f9'),
        legend=dict(font=dict(color='#f1f5f9')) )
    return fig
    

def create_match_gauge(match_score):
    # Create gauge without the number display
    fig = go.Figure(go.Indicator(
        mode="gauge",
        value=match_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {
                'range': [None, 100],
                'tickwidth': 1,
                'tickcolor': "#f1f5f9",
                'tickfont': {'size': 12}
            },
            'bar': {'color': "#3b82f6", 'thickness': 0.3},
            'bgcolor': "rgba(0,0,0,0)",
            'steps': [
                {'range': [0, 50], 'color': "#F87171"},
                {'range': [50, 80], 'color': "#FBBF24"},
                {'range': [80, 100], 'color': "#34D399"}
            ],
            'threshold': {
                'line': {'color': "#f1f5f9", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    # Add title as annotation above the gauge
    fig.add_annotation(
        x=0.5,
        y=1.15,
        xref="paper",
        yref="paper",
        text="Overall Match Score",
        showarrow=False,
        font=dict(size=18, color='#f1f5f9'),
        align="center"
    )
    
    # Add match score precisely at the end of the rainbow structure
    fig.add_annotation(
        x=0.5,
        y=0.10,  # Center of the gauge
        xref="paper",
        yref="paper",
        text=f"{match_score:.1f}%",
        showarrow=False,
        font=dict(size=40, color='#f1f5f9'),
        align="center",
        # Position slightly below the pointer
        yshift=-20  # Move down 20 pixels from center
    )
    
    fig.update_layout(
        height=350,
        paper_bgcolor='#0f172a',
        font={'color': "#f1f5f9"},
        margin=dict(t=60, b=60, l=20, r=20),
        showlegend=False
    )
    return fig

def create_role_prediction_chart(roles, scores):
    if not roles or not scores:
        return None

    df = pd.DataFrame({
        'Role': roles[:6],
        'Confidence': scores[:6]
    })

    fig = px.bar(
        df,
        x='Confidence',
        y='Role',
        orientation='h',
        title="Job Role Predictions",
        color='Confidence',
        color_continuous_scale='Viridis',
        text='Confidence'
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        height=400,
        yaxis={'categoryorder': 'total ascending', 'showgrid': False, 'color': '#f1f5f9'},
        plot_bgcolor='#0f172a',  # Changed to dark background
        paper_bgcolor='#0f172a',  # Changed to dark background
        font={'color': '#f1f5f9'},
        title_font={'size': 18, 'color': '#f1f5f9'},
        xaxis=dict(showgrid=False, color='#f1f5f9')
    )
    return fig

def format_experience(level):
    """Show experience as-is if present; else 'Fresher'."""
    if not level or str(level).strip().lower() in ['not mentioned', 'not specified', 'none', '']:
        return "Fresher"
    else:
        return str(level)
        
def format_education(education):
    """Extract degree from education information"""
    if not education:
        return "Not Mentioned"
    
    # Handle list format
    if isinstance(education, list):
        # Take the first education entry
        if not education: return "Not Mentioned"
        education_str = education[0]
    else:
        education_str = education
        
    # Common degree patterns
    degree_patterns = [
        r'B\.?Tech\b', r'B\.?E\.?\b', r'B\.?Sc\b', r'B\.?Com\b', r'B\.?C\.?A\b', r'B\.?C\.?S\b',
        r'M\.?Tech\b', r'M\.?E\.?\b', r'M\.?Sc\b', r'M\.?Com\b', r'M\.?C\.?A\b', r'M\.?C\.?S\b',
        r'Ph\.?D\b', r'Doctorate\b', r'Bachelor\b', r'Master\b'
    ]
    
    # Try to find a degree match
    for pattern in degree_patterns:
        match = re.search(pattern, education_str, re.IGNORECASE)
        if match:
            return match.group(0).upper()
            
    # If no pattern matched, return first 3 words
    words = education_str.split()[:3]
    return " ".join(words)

def metric_card(label, value):
    return f"""
    <div class="metric-card">
        <div style="font-size: 1rem; color: #94a3b8; font-weight: 600; margin-bottom: 4px;">{label}</div>
        <div style="font-size: 1.25rem; color: #f1f5f9; font-weight: 700; word-wrap: break-word;">{value}</div>
    </div>
    """

def display_parsed_data_enhanced(parsed_data):
    if not parsed_data:
        st.info("No resume data parsed.")
        return

    st.subheader("📄 Resume Analysis Overview")

    # Personal Information Section
    st.markdown("### 👤 Personal Information")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        name = parsed_data.get('name', 'Not Mentioned')
        st.markdown(metric_card("Full Name", name), unsafe_allow_html=True)
    with col2:
        email = parsed_data.get('email', 'Not Mentioned')
        st.markdown(metric_card("Email Address", email), unsafe_allow_html=True)
    with col3:
        phone = parsed_data.get('phone', 'Not Mentioned')
        st.markdown(metric_card("Phone Number", phone), unsafe_allow_html=True)
    with col4:
        linkedin = parsed_data.get('linkedin', 'Not Mentioned')
        st.markdown(metric_card("LinkedIn", linkedin), unsafe_allow_html=True)

    # Professional Information
    st.markdown("### 💼 Professional Information")
    col4, col5 = st.columns(2)

    with col4:
        experience = parsed_data.get('experience')
        experience_display = format_experience(experience)
        st.markdown(metric_card("Experience Level", experience_display), unsafe_allow_html=True)

    with col5:
        education = parsed_data.get('education', ['Not Mentioned'])
        education_display = format_education(education)
        st.markdown(metric_card("Education", education_display), unsafe_allow_html=True)

    # Skills Section with tabs (List first, then distribution; skill cloud removed)
    skills = parsed_data.get('skills', [])
    st.markdown("### 🛠 Skills Analysis")
    if skills:
        tab1, tab2 = st.tabs(["Skills List", "Skills Distribution"])
        with tab1:
            st.markdown("#### Top Skills")
            cols = st.columns(4)
            for i, skill in enumerate(skills[:20]):
                with cols[i % 4]:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
                                color: white; padding: 8px 15px; border-radius: 10px;
                                margin: 5px; text-align: center; font-size: 0.9rem; word-break: break-word;'>
                        {skill}
                    </div>
                    """, unsafe_allow_html=True)
        with tab2:
            pie_fig = create_skills_pie_chart(skills)
            if pie_fig:
                st.plotly_chart(pie_fig, use_container_width=True)
    else:
        st.info("No specific skills were extracted from the resume.")

# Function to create download links
def create_download_link(data, filename, text):
    """Create a download link for files"""
    b64 = base64.b64encode(data.encode('utf-8')).decode() # Use utf-8 for HTML
    href = f'<a class="download-btn" href="data:file/html;base64,{b64}" download="{filename}">{text}</a>'
    return href

def create_pdf_download_link(data, filename, text):
    """Create a download link for PDF files"""
    b64 = base64.b64encode(data).decode()
    href = f'<a class="download-btn" href="data:application/pdf;base64,{b64}" download="{filename}">{text}</a>'
    return href

# ==============================================================================
#                      ADVANCED ATS SCORE CALCULATOR (INDUSTRY-STANDARD)
# ==============================================================================
def calculate_ats_score(parsed_info, job_description, resume_text):
    """
    Calculate ATS compatibility score using industry best practices
    Based on standards from Jobscan, ResumeWorded, and TopResume
    
    Scoring Categories:
    1. Contact Information (5 points)
    2. Experience & Education (20 points)
    3. Skills & Keywords (30 points)
    4. Formatting & Structure (20 points)
    5. Achievements & Quantifiable Results (15 points)
    6. Customization (10 points)
    """
    score = 0
    breakdown = {}
    recommendations = []
    matched_keywords = []
    missing_keywords = []
    
    # 1. Contact Information (5 points)
    contact_score = 0
    if parsed_info.get('name') not in [None, 'Not Mentioned']: 
        contact_score += 1
    if parsed_info.get('email') not in [None, 'Not Mentioned']: 
        contact_score += 1
    if parsed_info.get('phone') not in [None, 'Not Mentioned']: 
        contact_score += 1
    if parsed_info.get('linkedin') not in [None, 'Not Mentioned']: 
        contact_score += 1
    if contact_score < 3:
        recommendations.append("Add missing contact information (name, email, phone, LinkedIn)")
    if contact_score == 4:
        contact_score += 1  # Bonus point for complete contact info
        
    score += contact_score
    breakdown['Contact Info'] = contact_score
    
    # 2. Experience & Education (20 points)
    exp_edu_score = 0
    experience = parsed_info.get('experience')
    if experience not in [None, 'Not Mentioned']: 
        exp_edu_score += 8
        # Check for quantifiable achievements
        if re.search(r'\d+', str(experience)):  # Check for numbers in experience
            exp_edu_score += 2
        else:
            recommendations.append("Add quantifiable achievements (numbers, percentages) to your experience section")
    else:
        recommendations.append("Add work experience section with specific details")
        
    education = parsed_info.get('education', ['Not Mentioned'])
    if education[0] != 'Not Mentioned': 
        exp_edu_score += 8
        # Check for degree details
        if any(word in education[0].lower() for word in ['bachelor', 'master', 'phd', 'associate', 'diploma']):
            exp_edu_score += 2
    else:
        recommendations.append("Add education details including degree and institution")
        
    score += exp_edu_score
    breakdown['Experience & Education'] = exp_edu_score
    
    # 3. Skills & Keywords (30 points) - Most important section
    skills_score = 0
    skills = parsed_info.get('skills', [])
    if skills:
        skills_score += 10
        
        # Skill count
        if len(skills) >= 10:
            skills_score += 5
        elif len(skills) >= 5:
            skills_score += 3
        else:
            recommendations.append("Add more skills (at least 5-10 relevant ones)")
            
        # Job description keyword matching
        if job_description:
            # Extract keywords from job description
            job_desc = job_description.lower()
            resume = resume_text.lower()
            
            # Find required skills/qualifications section
            required_section = ""
            required_patterns = [
                r'requirements?:?(.*?)(?=desired|preferred|responsibilities|qualifications|$)', 
                r'qualifications?:?(.*?)(?=desired|preferred|responsibilities|requirements|$)',
                r'skills?:?(.*?)(?=desired|preferred|responsibilities|requirements|qualifications|$)'
            ]
            
            for pattern in required_patterns:
                match = re.search(pattern, job_desc, re.IGNORECASE | re.DOTALL)
                if match:
                    required_section = match.group(1)
                    break
            
            # If no specific section found, use entire JD
            if not required_section:
                required_section = job_desc
                
            # Extract keywords from required section
            words = re.findall(r'\b\w{4,}\b', required_section)
            stop_words = set(stopwords.words('english'))
            words = [word for word in words if word not in stop_words]
            freq = Counter(words)
            top_keywords = [word for word, count in freq.most_common(25)]
            
            # Check for matches in skills and experience
            matched_keywords = [kw for kw in top_keywords if kw in resume]
            missing_keywords = [kw for kw in top_keywords if kw not in resume]
            
            # Score based on match percentage
            if top_keywords:
                match_percent = len(matched_keywords) / len(top_keywords)
                skills_score += match_percent * 15
                
                # Add recommendations for missing keywords
                if missing_keywords:
                    recommendations.append(f"Add these keywords to your resume: {', '.join(missing_keywords[:5])}...")
    else:
        recommendations.append("Add a dedicated skills section with relevant technical and soft skills")
        
    score += skills_score
    breakdown['Skills & Keywords'] = skills_score
    
    # 4. Formatting & Structure (20 points)
    format_score = 0
    # Check for section headings
    section_headings = ['experience', 'education', 'skills', 'projects', 'certifications']
    resume_lower = resume_text.lower()
    present_sections = 0
    
    for section in section_headings:
        if section in resume_lower:
            present_sections += 1
            format_score += 2
    
    # Check for professional summary
    if any(word in resume_lower for word in ['summary', 'objective', 'profile']):
        format_score += 2
    else:
        recommendations.append("Add a professional summary section at the top")
    
    # Check length
    word_count = len(resume_text.split())
    if 500 <= word_count <= 800:  # Ideal resume length
        format_score += 4
    elif word_count < 400:
        recommendations.append(f"Resume is too short ({word_count} words), add more details")
    else:
        recommendations.append(f"Resume is too long ({word_count} words), condense to 500-800 words")
    
    # Check bullet points
    bullet_points = resume_text.count('•') + resume_text.count('-') + resume_text.count('*')
    if bullet_points >= 5:
        format_score += 4
    else:
        recommendations.append("Use bullet points for achievements and responsibilities")
        
    # Check readability
    try:
        readability = textstat.flesch_reading_ease(resume_text)
        if readability >= 60:  # Fairly easy to read
            format_score += 2
    except:
        pass
        
    score += format_score
    breakdown['Formatting & Structure'] = format_score
    
    # 5. Achievements & Quantifiable Results (15 points)
    achievements_score = 0
    # Count numbers in resume (indicating quantifiable results)
    numbers = re.findall(r'\b\d+\b', resume_text)
    if len(numbers) >= 3:
        achievements_score += 10
    elif len(numbers) >= 1:
        achievements_score += 5
    else:
        recommendations.append("Add quantifiable achievements (e.g., 'increased sales by 20%')")
    
    # Check for action verbs
    action_verbs = ['achieved', 'managed', 'developed', 'led', 'increased', 'reduced', 
                   'improved', 'created', 'implemented', 'spearheaded']
    action_count = sum(1 for verb in action_verbs if verb in resume_lower)
    if action_count >= 5:
        achievements_score += 5
    elif action_count >= 3:
        achievements_score += 3
        
    score += achievements_score
    breakdown['Achievements'] = achievements_score
    
    # 6. Customization (10 points) - How well resume matches job description
    customization_score = 0
    if job_description:
        # Calculate similarity between resume and job description
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        customization_score = min(similarity * 10, 10)
        
        if customization_score < 7:
            recommendations.append("Customize your resume more specifically for this job")
    else:
        customization_score = 5  # Base score when no JD provided
    
    score += customization_score
    breakdown['Customization'] = customization_score
    
    # Cap score at 100
    score = min(score, 100)
    
    return {
        'score': score,
        'breakdown': breakdown,
        'recommendations': recommendations,
        'matched_keywords': matched_keywords,
        'missing_keywords': missing_keywords
    }

def display_ats_score(ats_result):
    """Display ATS score with detailed breakdown and recommendations"""
    score = ats_result['score']
    breakdown = ats_result['breakdown']
    recommendations = ats_result['recommendations']
    matched_keywords = ats_result.get('matched_keywords', [])
    missing_keywords = ats_result.get('missing_keywords', [])
    
    # Determine color based on score
    if score >= 90:
        color = "#34D399"  # Green
        status = "Excellent"
        feedback = "Your resume has excellent ATS compatibility and is well-optimized"
    elif score >= 80:
        color = "#A3E635"  # Light green
        status = "Very Good"
        feedback = "Your resume has very good ATS compatibility with minor improvements possible"
    elif score >= 70:
        color = "#FBBF24"  # Yellow
        status = "Good"
        feedback = "Your resume has good compatibility but could be improved"
    elif score >= 60:
        color = "#FB923C"  # Orange
        status = "Fair"
        feedback = "Your resume has fair compatibility and needs optimization"
    else:
        color = "#F87171"  # Red
        status = "Needs Improvement"
        feedback = "Your resume needs significant improvement to pass ATS screening"
    
    # Create gauge-like visualization
    st.markdown(f"""
    <div class="ats-score-card">
        <div class="ats-score-value" style="color: {color};">{score:.0f}/100</div>
        <div class="ats-score-label">ATS Compatibility Score: {status}</div>
        <div style="text-align: center; margin-top: 10px; color: #94a3b8;">{feedback}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Score breakdown
    st.markdown("#### 📈 Score Breakdown")
    st.markdown("<div class='score-breakdown'>", unsafe_allow_html=True)
    for category, value in breakdown.items():
        st.markdown(f"""
        <div class="score-category">
            <div class="score-category-name">{category}</div>
            <div class="score-category-value">{value:.1f}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Keyword analysis
    if matched_keywords or missing_keywords:
        st.markdown("#### 🔑 Keyword Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"✅ {len(matched_keywords)} Keywords Matched")
            if matched_keywords:
                st.markdown("##### Present in your resume:")
                cols = st.columns(3)
                for i, keyword in enumerate(matched_keywords[:9]):
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
                                    color: white; padding: 6px 12px; border-radius: 8px;
                                    margin: 5px; text-align: center; font-size: 0.8rem;'>
                            {keyword}
                        </div>
                        """, unsafe_allow_html=True)
        
        with col2:
            st.error(f"⚠️ {len(missing_keywords)} Keywords Missing")
            if missing_keywords:
                st.markdown("##### Should be added:")
                cols = st.columns(3)
                for i, keyword in enumerate(missing_keywords[:9]):
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #F87171 0%, #FB7185 100%);
                                    color: white; padding: 6px 12px; border-radius: 8px;
                                    margin: 5px; text-align: center; font-size: 0.8rem;'>
                            {keyword}
                        </div>
                        """, unsafe_allow_html=True)
    
    # Recommendations
    st.markdown("#### 🔍 Recommendations to Improve:")
    if recommendations:
        for rec in recommendations:
            st.markdown(f"""
            <div class="recommendation-box">
                ✅ {rec}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="recommendation-box">
            ✅ Your resume has excellent ATS compatibility
        </div>
        <div class="recommendation-box">
            ✅ Maintain current formatting and content structure
        </div>
        <div class="recommendation-box">
            ✅ Continue tailoring for specific roles
        </div>
        """, unsafe_allow_html=True)
    
    return score

# ==============================================================================
#                      OPTIMIZED ATS-FRIENDLY RESUME GENERATOR
# ==============================================================================
def create_ats_friendly_pdf(parsed_info, skills_to_learn, job_title=None):
    """Create a visually impressive, modern ATS-friendly resume PDF with a premium look."""
    def clean_text(text):
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        return text.encode('latin-1', 'ignore').decode('latin-1')

    pdf = FPDF()
    pdf.add_page()

    # --- HEADER with gradient effect ---
    pdf.set_fill_color(59, 130, 246)  # Blue
    pdf.rect(0, 0, 210, 38, 'F')
    pdf.set_font("Helvetica", 'B', 28)
    pdf.set_text_color(255, 255, 255)
    name = clean_text(parsed_info.get('name', 'Your Name'))
    pdf.cell(0, 18, name, ln=1, align='C')
    pdf.set_font("Helvetica", '', 12)
    pdf.set_text_color(230, 245, 255)
    email = clean_text(parsed_info.get('email', 'email@example.com'))
    phone = clean_text(parsed_info.get('phone', '123-456-7890'))
    linkedin = clean_text(parsed_info.get('linkedin', ''))
    contact_info = f"{email} | {phone}"
    if linkedin:
        contact_info += f" | LinkedIn: {linkedin}"
    pdf.cell(0, 10, contact_info, ln=1, align='C')
    pdf.ln(2)

    # --- PROFILE SUMMARY with highlight box ---
    pdf.set_xy(10, 45)
    pdf.set_fill_color(236, 245, 255)
    pdf.set_draw_color(59, 130, 246)
    pdf.set_line_width(0.5)
    pdf.rect(10, pdf.get_y(), 190, 22, 'DF')
    pdf.set_xy(12, pdf.get_y() + 2)
    pdf.set_font("Helvetica", 'B', 14)
    pdf.set_text_color(59, 130, 246)
    pdf.cell(0, 8, "Professional Summary", ln=1)
    pdf.set_font("Helvetica", '', 11)
    pdf.set_text_color(30, 41, 59)
    summary = parsed_info.get('summary')
    if not summary:
        experience_level = clean_text(format_experience(parsed_info.get('experience')))
        job_title_clean = clean_text(job_title or parsed_info.get('target_role', 'target position'))
        skills = parsed_info.get('skills', [])
        top_skills = ', '.join(skills[:5]) if skills else "technical and soft skills"
        summary = (f"Dynamic {experience_level} professional skilled in {top_skills}. "
                   f"Seeking a {job_title_clean} role to leverage proven results and drive success.")
    pdf.set_x(12)
    pdf.multi_cell(186, 7, clean_text(summary), align='L')
    pdf.ln(2)

    # --- SKILLS with colored badges ---
    pdf.set_font("Helvetica", 'B', 13)
    pdf.set_text_color(59, 130, 246)
    pdf.cell(0, 8, "Key Skills", ln=1)
    pdf.set_font("Helvetica", '', 11)
    pdf.set_text_color(30, 41, 59)
    skills = parsed_info.get('skills', [])
    if skills:
        x_start = pdf.get_x()
        for i, skill in enumerate(skills[:12]):
            if i % 4 == 0 and i != 0:
                pdf.ln(8)
                pdf.set_x(x_start)
            pdf.set_fill_color(99, 102, 241)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(42, 8, clean_text(skill), border=0, ln=0, align='C', fill=True)
            pdf.set_text_color(30, 41, 59)
            pdf.cell(3, 8, "", border=0, ln=0)
        pdf.ln(12)
    else:
        pdf.cell(0, 8, "Add more relevant skills to stand out.", ln=1)
    pdf.ln(2)

    # --- EXPERIENCE with timeline style and modern look ---
    pdf.set_font("Helvetica", 'B', 13)
    pdf.set_text_color(59, 130, 246)
    pdf.cell(0, 8, "Professional Experience", ln=1)
    pdf.set_font("Helvetica", '', 11)
    pdf.set_text_color(30, 41, 59)
    experience_data = parsed_info.get('experience')
    if experience_data:
        if isinstance(experience_data, list):
            for exp in experience_data[:2]:
                pdf.set_font("Helvetica", 'B', 12)
                pdf.set_text_color(99, 102, 241)
                pdf.cell(0, 7, clean_text(exp), ln=1)
                pdf.set_font("Helvetica", '', 11)
                pdf.set_text_color(30, 41, 59)
                pdf.cell(0, 7, "- Achieved significant results in key projects", ln=1)
                pdf.cell(0, 7, "- Developed innovative solutions", ln=1)
                pdf.ln(1)
        else:
            pdf.set_font("Helvetica", 'B', 12)
            pdf.set_text_color(99, 102, 241)
            pdf.cell(0, 7, clean_text(experience_data), ln=1)
            pdf.set_font("Helvetica", '', 11)
            pdf.set_text_color(30, 41, 59)
            pdf.cell(0, 7, "- Led cross-functional teams to deliver projects", ln=1)
            pdf.cell(0, 7, "- Increased productivity by 40%", ln=1)
            pdf.ln(1)
    else:
        pdf.cell(0, 7, "Add your professional experience here.", ln=1)
    pdf.ln(2)

    # --- EDUCATION with highlight ---
    pdf.set_font("Helvetica", 'B', 13)
    pdf.set_text_color(59, 130, 246)
    pdf.cell(0, 8, "Education", ln=1)
    pdf.set_font("Helvetica", '', 11)
    pdf.set_text_color(30, 41, 59)
    education = parsed_info.get('education', ['Not Mentioned'])
    education_display = clean_text(format_education(education))
    pdf.cell(0, 7, education_display, ln=1)
    university = parsed_info.get('university', 'University of Excellence') or 'University of Excellence'
    pdf.cell(0, 7, clean_text(university), ln=1)
    pdf.ln(2)

    # --- CERTIFICATIONS & SPECIALIZED SKILLS ---
    pdf.set_font("Helvetica", 'B', 13)
    pdf.set_text_color(59, 130, 246)
    pdf.cell(0, 8, "Certifications & Specialized Skills", ln=1)
    pdf.set_font("Helvetica", '', 11)
    pdf.set_text_color(30, 41, 59)
    certifications = parsed_info.get('certifications', [])
    if certifications:
        for cert in certifications[:3]:
            pdf.cell(0, 7, f"- {clean_text(cert)}", ln=1)
    elif skills_to_learn:
        for skill in skills_to_learn[:3]:
            pdf.cell(0, 7, f"- {clean_text(skill)}", ln=1)
    else:
        pdf.cell(0, 7, "- Add certifications to boost your profile", ln=1)
    pdf.ln(2)

    # --- MODERN PROJECTS SECTION ---
    pdf.set_font("Helvetica", 'B', 13)
    pdf.set_text_color(59, 130, 246)
    pdf.cell(0, 8, "Key Projects", ln=1)
    pdf.set_font("Helvetica", '', 11)
    pdf.set_text_color(30, 41, 59)
    projects = parsed_info.get('projects', [])
    if projects:
        for project in projects[:2]:
            pdf.cell(0, 7, f"- {clean_text(project)}", ln=1)
    else:
        pdf.cell(0, 7, "- Add 1-2 key projects to showcase your work", ln=1)
    pdf.ln(2)

    # --- MODERN LANGUAGES SECTION ---
    pdf.set_font("Helvetica", 'B', 13)
    pdf.set_text_color(59, 130, 246)
    pdf.cell(0, 8, "Languages", ln=1)
    pdf.set_font("Helvetica", '', 11)
    pdf.set_text_color(30, 41, 59)
    languages = parsed_info.get('languages', [])
    if languages:
        pdf.cell(0, 7, ", ".join([clean_text(lang) for lang in languages]), ln=1)
    else:
        pdf.cell(0, 7, "English", ln=1)
    pdf.ln(2)

    # --- FOOTER ---
    pdf.set_y(-18)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(99, 102, 241)
    pdf.cell(0, 10, 'Generated by Resume Analyzer Pro | Modern ATS-Optimized Resume', 0, 0, 'C')

    return pdf.output(dest='S').encode('latin1')

# ==============================================================================
#                      ATTRACTIVE REPORT GENERATION
# ==============================================================================
def create_attractive_report(parsed_info, role_predictions, similarity_scores, action_items, skills_to_learn, job_title=None):
    """Create an attractive HTML report"""
    report_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Resume Analysis Report</title>
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Poppins', sans-serif;
                line-height: 1.6;
                color: #333;
                background: linear-gradient(135deg, #f5f7fa 0%, #e4e7eb 100%);
                padding: 30px;
            }}
            
            .container {{
                max-width: 1000px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                overflow: hidden;
                box-shadow: 0 15px 50px rgba(0, 0, 0, 0.1);
            }}
            
            .header {{
                background: linear-gradient(135deg, #2563EB 0%, #1D4ed8 100%);
                color: white;
                padding: 40px;
                text-align: center;
                position: relative;
                overflow: hidden;
            }}
            
            .header::before {{
                content: "";
                position: absolute;
                top: -50px;
                left: -50px;
                width: 200px;
                height: 200px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 50%;
            }}
            
            .header::after {{
                content: "";
                position: absolute;
                bottom: -80px;
                right: -80px;
                width: 250px;
                height: 250px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 50%;
            }}
            
            .header h1 {{
                font-size: 42px;
                font-weight: 700;
                margin-bottom: 10px;
                position: relative;
                z-index: 2;
            }}
            
            .header p {{
                font-size: 18px;
                opacity: 0.9;
                position: relative;
                z-index: 2;
            }}
            
            .badge {{
                display: inline-block;
                background: rgba(255, 255, 255, 0.2);
                padding: 8px 20px;
                border-radius: 50px;
                margin-top: 20px;
                font-size: 14px;
                position: relative;
                z-index: 2;
            }}
            
            .content {{
                padding: 40px;
            }}
            
            .section {{
                margin-bottom: 40px;
            }}
            
            .section-title {{
                font-size: 28px;
                color: #2563EB;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 3px solid #2563EB;
                display: inline-block;
            }}
            
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 25px;
                margin-bottom: 30px;
            }}
            
            .metric-card {{
                background: #f8fafc;
                border-radius: 15px;
                padding: 25px;
                text-align: center;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
                transition: all 0.3s ease;
                border: 1px solid #e2e8f0;
            }}
            
            .metric-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
                border-color: #cbd5e0;
            }}
            
            .metric-value {{
                font-size: 36px;
                font-weight: 700;
                color: #2563EB;
                margin: 15px 0;
            }}
            
            .metric-label {{
                font-size: 16px;
                color: #64748B;
            }}
            
            .skill-badge {{
                display: inline-block;
                background: linear-gradient(135deg, #3B82F6 0%, #6366F1 100%);
                color: white;
                padding: 8px 20px;
                border-radius: 50px;
                margin: 8px;
                font-size: 14px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                transition: all 0.3s ease;
            }}
            
            .skill-badge:hover {{
                transform: scale(1.05);
                box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15);
            }}
            
            .recommendation {{
                background: #EFF6FF;
                border-left: 4px solid #2563EB;
                padding: 25px;
                margin: 25px 0;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
            }}
            
            .recommendation h3 {{
                color: #2563EB;
                margin-bottom: 15px;
                font-size: 20px;
            }}
            
            .footer {{
                text-align: center;
                padding: 30px;
                background: #f1f5f9;
                color: #64748B;
                font-size: 14px;
                border-top: 1px solid #e2e8f0;
            }}
            
            .chart-container {{
                background: white;
                border-radius: 15px;
                padding: 25px;
                margin: 25px 0;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
            }}
            
            .role-card {{
                background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                border-radius: 15px;
                padding: 25px;
                margin: 20px 0;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
                border: 1px solid #e2e8f0;
            }}
            
            .role-title {{
                font-size: 22px;
                color: #2563EB;
                margin-bottom: 10px;
            }}
            
            .role-score {{
                font-size: 28px;
                font-weight: 700;
                color: #10B981;
                margin: 10px 0;
            }}
            
            .role-desc {{
                color: #64748B;
                margin-top: 15px;
            }}
            
            .progress-container {{
                margin: 20px 0;
            }}
            
            .progress-bar {{
                height: 12px;
                background: #e2e8f0;
                border-radius: 10px;
                overflow: hidden;
                margin: 10px 0;
            }}
            
            .progress-fill {{
                height: 100%;
                background: linear-gradient(135deg, #10B981 0%, #34D399 100%);
                border-radius: 10px;
            }}
            
            .insight-card {{
                background: white;
                border-radius: 15px;
                padding: 25px;
                margin: 20px 0;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
                border-left: 4px solid #2563EB;
            }}
            
            .insight-title {{
                color: #2563EB;
                font-size: 20px;
                margin-bottom: 15px;
            }}
            
            .match-level {{
                font-size: 24px;
                font-weight: 700;
                text-align: center;
                padding: 15px;
                border-radius: 12px;
                margin: 20px 0;
            }}
            
            .excellent {{
                background: linear-gradient(135deg, #34D399 0%, #10B981 100%);
                color: white;
            }}
            
            .good {{
                background: linear-gradient(135deg, #FBBF24 0%, #F59E0B 100%);
                color: white;
            }}
            
            .moderate {{
                background: linear-gradient(135deg, #F87171 0%, #EF4444 100%);
                color: white;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Resume Analysis Report</h1>
                <p>Generated on {datetime.now().strftime('%B %d, %Y')}</p>
                <div class="badge">Professional Analysis</div>
            </div>
            
            <div class="content">
                <div class="section">
                    <h2 class="section-title">Candidate Overview</h2>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value">{parsed_info.get('name', 'Not Mentioned')}</div>
                            <div class="metric-label">Full Name</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{format_experience(parsed_info.get('experience'))}</div>
                            <div class="metric-label">Experience Level</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{len(parsed_info.get('skills', []))}</div>
                            <div class="metric-label">Skills Identified</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{role_predictions.get('roles', ['Not Detected'])[0]}</div>
                            <div class="metric-label">Top Role Match</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2 class="section-title">Role Predictions</h2>
                    <div class="metrics-grid">
    """
    
    # Add role predictions
    if role_predictions.get('roles'):
        for role, score in zip(role_predictions['roles'][:4], role_predictions['scores'][:4]):
            report_html += f"""
                        <div class="metric-card">
                            <div class="metric-value">{score:.1f}%</div>
                            <div class="metric-label">{role}</div>
                        </div>
            """
    
    report_html += """
                    </div>
                    
                    <div class="chart-container">
                        <h3>Detailed Role Analysis</h3>
    """
    
    if role_predictions.get('roles'):
        for i, (role, score) in enumerate(zip(role_predictions['roles'][:3], role_predictions['scores'][:3])):
            report_html += f"""
                        <div class="role-card">
                            <div class="role-title">{i+1}. {role}</div>
                            <div class="role-score">{score:.1f}% Match</div>
                            <div class="progress-container">
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: {score}%"></div>
                                </div>
                            </div>
                            <div class="role-desc">
                                This role aligns well with your skills and experience. 
                                Consider highlighting your expertise in relevant areas.
                            </div>
                        </div>
            """
    
    report_html += """
                    </div>
                </div>
    """
    
    # Add matching analysis if available
    if job_title and similarity_scores:
        overall_match = similarity_scores.get('combined_score', 0.0)
        
        # Determine match level
        if overall_match >= 90:
            match_level = "Perfect Match"
            match_class = "excellent"
        elif overall_match >= 80:
            match_level = "Excellent Match"
            match_class = "excellent"
        elif overall_match >= 70:
            match_level = "Strong Match"
            match_class = "good"
        elif overall_match >= 60:
            match_level = "Good Match"
            match_class = "good"
        elif overall_match >= 40:
            match_level = "Moderate Match"
            match_class = "moderate"
        else:
            match_level = "Low Match"
            match_class = "moderate"
        
        report_html += f"""
                <div class="section">
                    <h2 class="section-title">Job Matching: {job_title}</h2>
                    
                    <div class="insight-card">
                        <div class="match-level {match_class}">{match_level}</div>
                        <p>Your resume has an overall match score of <strong>{overall_match:.1f}%</strong> with the job description.</p>
                    </div>
                    
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value">{similarity_scores.get('tfidf_similarity', 0.0):.1f}%</div>
                            <div class="metric-label">Content Match</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{similarity_scores.get('keyword_similarity', 0.0):.1f}%</div>
                            <div class="metric-label">Keyword Match</div>
                        </div>
                    </div>
        """
        
        # Add matching insights
        report_html += """
                    <div class="insight-card">
                        <div class="insight-title">Matching Insights</div>
        """
        
        if overall_match >= 90:
            report_html += """
                        <p>🎉 Your resume shows perfect alignment with the job requirements. You're an ideal candidate for this position!</p>
                        <p><strong>Recommendations:</strong></p>
                        <ul>
                            <li>Highlight your top matching skills at the beginning of your resume</li>
                            <li>Quantify achievements related to the job requirements</li>
                            <li>Prepare specific examples of your relevant experience</li>
                        </ul>
            """
        elif overall_match >= 80:
            report_html += """
                        <p>🌟 Your resume shows excellent alignment with the job requirements. You're a strong candidate for this position!</p>
                        <p><strong>Recommendations:</strong></p>
                        <ul>
                            <li>Emphasize your most relevant experience</li>
                            <li>Include specific metrics to quantify your impact</li>
                            <li>Highlight certifications that match the job</li>
                        </ul>
            """
        elif overall_match >= 70:
            report_html += """
                        <p>👍 Your resume has strong alignment with the job requirements. With minor improvements, you'll be highly competitive.</p>
                        <p><strong>Recommendations:</strong></p>
                        <ul>
                            <li>Add 1-2 more keywords from the job description</li>
                            <li>Emphasize transferable skills</li>
                            <li>Include specific metrics to quantify your achievements</li>
                        </ul>
            """
        elif overall_match >= 60:
            report_html += """
                        <p>⚠️ Your resume shows good alignment with the job requirements. Consider these improvements to be more competitive.</p>
                        <p><strong>Recommendations:</strong></p>
                        <ul>
                            <li>Add 3-5 more keywords from the job description</li>
                            <li>Include more metrics to quantify achievements</li>
                            <li>Highlight relevant projects</li>
                        </ul>
            """
        elif overall_match >= 40:
            report_html += """
                        <p>📝 Your resume has moderate alignment with the job requirements. Focus on developing these areas.</p>
                        <p><strong>Recommendations:</strong></p>
                        <ul>
                            <li>Focus on developing skills in gap areas</li>
                            <li>Consider taking relevant courses or certifications</li>
                            <li>Tailor your resume specifically for this role</li>
                        </ul>
            """
        else:
            report_html += """
                        <p>❌ Your resume has low alignment with the job requirements. Consider these strategies.</p>
                        <p><strong>Recommendations:</strong></p>
                        <ul>
                            <li>Re-evaluate if this role aligns with your career goals</li>
                            <li>Consider similar but more junior positions</li>
                            <li>Focus on developing the required skills</li>
                        </ul>
            """
        
        report_html += """
                    </div>
                </div>
        """
    
    # Add skills analysis
    report_html += """
                <div class="section">
                    <h2 class="section-title">Skills Analysis</h2>
                    <h3>Top Skills</h3>
    """
    
    if parsed_info.get('skills'):
        for skill in parsed_info.get('skills')[:15]:
            report_html += f'<span class="skill-badge">{skill}</span>'
    
    if skills_to_learn:
        report_html += """
                    <h3 style="margin-top: 30px;">Recommended Skills</h3>
        """
        for skill in skills_to_learn[:12]:
            report_html += f'<span class="skill-badge" style="background: linear-gradient(135deg, #10B981 0%, #34D399 100%);">{skill}</span>'
    
    # Add recommendations
    report_html += """
                </div>
                
                <div class="section">
                    <h2 class="section-title">Recommendations</h2>
    """
    
    for item in action_items:
        report_html += f'<div class="recommendation">{item}</div>'
    
    if skills_to_learn:
        report_html += f"""
                    <div class="recommendation">
                        <h3>Skill Development Roadmap</h3>
                        <p>Focus on acquiring these skills to increase your competitiveness for the target role:</p>
                        <ul>
        """
        for skill in skills_to_learn[:8]:
            report_html += f'<li>{skill} - Consider online courses or certifications</li>'
        report_html += "</ul></div>"
    
    report_html += """
                </div>
            </div>
            
            <div class="footer">
                <p>Generated by Resume Analyzer Pro | Professional Resume Analysis</p>
                <p>This report is designed to help you optimize your resume and stand out to recruiters.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return report_html

# Main UI Title and Date with Professional Styling
st.markdown("""
<div style="max-width: 1200px; margin: 0 auto; padding: 40px 20px;">
  <h1 class="main-title">🚀 Resume Analyzer Pro</h1>
  <p style="text-align: center; color: #94a3b8; font-size: 18px; margin-top: 0;">
    Advanced ML-Powered Resume Analysis | """ + datetime.now().strftime('%B %d, %Y') + """
  </p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    if st.session_state.get("logged_in"):
        st.markdown(f"### 👤 Welcome, {st.session_state['user_email']}")
    
    st.markdown("### 🎛️ Control Panel")

    st.markdown("#### 🛠 System Status")
    if jms_system.clf_model and jms_system.tfidf and jms_system.encoder:
        st.success("✅ All ML Models Loaded Successfully")
        st.info("📊 Ready for Advanced Analysis")
    else:
        missing_models = []
        if not jms_system.clf_model: missing_models.append("Classifier")
        if not jms_system.tfidf: missing_models.append("TF-IDF Vectorizer")
        if not jms_system.encoder: missing_models.append("Label Encoder")
        st.warning(f"⚠️ Missing Models: {', '.join(missing_models)}")

    st.markdown("---")

    st.markdown("#### 📁 Upload Resume")
    uploaded_file = st.file_uploader(
        "Choose your resume file",
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, DOCX, TXT"
    )

    st.markdown("#### 📄 Job Description (Optional)")
    job_description = st.text_area(
        "Paste job description for matching analysis",
        height=150,
        placeholder="Paste the job description here to get similarity scores and better recommendations...",
        key="job_desc"
    )

    st.markdown("#### 🚀 Analysis")
    analyze_button = st.button(
        "🔬 Start Deep Analysis",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_file
    )

    if uploaded_file:
        st.success(f"✅ File '{uploaded_file.name}' ready for analysis")
        
# Main Content Area with Professional Styling
if analyze_button and uploaded_file:
    resume_text = ""
    try:
        with st.spinner("📖 Reading resume file..."):
            if uploaded_file.name.endswith('.pdf'):
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    resume_text += page.extract_text() or ''
            elif uploaded_file.name.endswith('.docx'):
                doc = docx.Document(uploaded_file)
                resume_text = "\n".join([para.text for para in doc.paragraphs])
            elif uploaded_file.name.endswith('.txt'):
                resume_text = uploaded_file.getvalue().decode("utf-8", errors='ignore')
            else:
                st.error("❌ Unsupported file type.")
                st.stop()

            if not resume_text.strip():
                st.warning("⚠️ Could not extract text from the uploaded resume. It might be an image-based PDF or empty file.")
                st.stop()

    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        st.stop()

    with st.spinner("🧠 Performing deep ML analysis... This may take a moment."):
        analysis_results = jms_system.analyze_resume_complete(
            resume_text,
            job_description=job_description if job_description.strip() else None
        )

    st.balloons()
    st.success("🎉 Analysis completed successfully!")

    if "error" in analysis_results:
        st.error(f"❌ Analysis Error: {analysis_results['error']}")
    else:
        st.markdown("---")

        parsed_info = analysis_results.get('parsed_resume', {})
        role_predictions = analysis_results.get('role_predictions', {})
        top_role = role_predictions.get('roles', ["Not Detected"])[0]
        top_role_score = role_predictions.get('scores', [0])[0]
        similarity_scores = analysis_results.get('similarity_scores', {})
        job_title = None
        
        if job_description:
            # Extract job title from job description
            if "job_title" in analysis_results:
                job_title = analysis_results["job_title"]
            else:
                # Simple extraction from first line
                job_title = job_description.split('\n')[0][:50]

        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

        metric_label_style = "font-size: 0.9rem; color: #94a3b8; font-weight: 600;"
        metric_value_style = "font-size: 1.15rem; color: #f1f5f9; font-weight: 700;"

        with summary_col1:
            candidate_name = parsed_info.get('name', 'Not Mentioned')
            st.markdown(f"""
            <div>
                <div style="{metric_label_style} margin-bottom:4px;">👤 Candidate</div>
                <div style="{metric_value_style}">{candidate_name}</div>
            </div>
            """, unsafe_allow_html=True)

        with summary_col2:
            st.markdown(f"""
            <div>
                <div style="{metric_label_style} margin-bottom:4px;">🎯 Top Role</div>
                <div style="{metric_value_style}">{top_role}</div>
                <div style="color:#94a3b8;">{top_role_score:.1f}% confidence</div>
            </div>
            """, unsafe_allow_html=True)

        with summary_col3:
            if job_description.strip() and similarity_scores:
                overall_match = similarity_scores.get('combined_score', 0.0)

                st.markdown(f"""
            <div>
                <div style="{metric_label_style} margin-bottom:4px;">🔗 JD Match</div>
                <div style="{metric_value_style}">{overall_match:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
            <div>
                <div style="{metric_label_style}">🔗 JD Match</div>
                <div style="{metric_value_style}">No JD provided</div>
            </div>
            """, unsafe_allow_html=True)

        with summary_col4:
            skills_count = len(parsed_info.get('skills', []))
            st.markdown(f"""
            <div>
                <div style="{metric_label_style} margin-bottom:4px;">🛠 Skills Found</div>
                <div style="{metric_value_style}">{skills_count}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        tab1, tab2, tab3, tab4 = st.tabs(["📄 Resume Analysis", "🎯 Role Predictions", "🔗 JD Matching", "💼 Job Recommendations"])

        with tab1:
            display_parsed_data_enhanced(parsed_info)
            
            # ATS Score Checker
            st.markdown("### 🎯 ATS Compatibility Score")
            ats_result = calculate_ats_score(parsed_info, job_description, resume_text)
            ats_score = display_ats_score(ats_result)

            # Additional resume insights
            st.markdown("### 📊 Resume Quality Insights")
            col_insights1, col_insights2 = st.columns(2)

            with col_insights1:
                completeness_factors = {
                    'Name': 1 if parsed_info.get('name') not in [None, 'Not Mentioned'] else 0,
                    'Email': 1 if parsed_info.get('email') not in [None, 'Not Mentioned'] else 0,
                    'Phone': 1 if parsed_info.get('phone') not in [None, 'Not Mentioned'] else 0,
                    'LinkedIn': 1 if parsed_info.get('linkedin') not in [None, 'Not Mentioned'] else 0,
                    'Experience': 1 if parsed_info.get('experience') not in [None, 'Not Mentioned'] else 0,
                    'Education': 1 if parsed_info.get('education', ['Not Mentioned'])[0] != 'Not Mentioned' else 0,
                    'Skills': 1 if len(parsed_info.get('skills', [])) > 0 else 0
                }

                completeness_score = (sum(completeness_factors.values()) / len(completeness_factors)) * 100
                completeness_df = pd.DataFrame(list(completeness_factors.items()), columns=['Field', 'Present'])
                completeness_df['Status'] = completeness_df['Present'].map({1: 'Found', 0: 'Missing'})

                fig_completeness = px.pie(
                    completeness_df,
                    names='Field',
                    values='Present',
                    color='Status',
                    title=f"Resume Completeness: {completeness_score:.0f}%",
                    color_discrete_map={'Found': '#34D399', 'Missing': '#F87171'},
                    hole=0.4
                )
                fig_completeness.update_layout(
                    plot_bgcolor='#0f172a',
                    paper_bgcolor='#0f172a',
                    font={'color': '#f1f5f9'},
                    legend=dict(font=dict(color='#f1f5f9')),
                    title_font=dict(color='#f1f5f9')
                )
                st.plotly_chart(fig_completeness, use_container_width=True)

            with col_insights2:
                word_count = len(resume_text.split())
                char_count = len(resume_text)
                st.markdown("#### 📈 Text Statistics")
                st.metric("Word Count", f"{word_count:,}")
                st.metric("Character Count", f"{char_count:,}")
                if word_count < 500:
                    length_assessment = "Too Short (Add more details)"
                    length_color = "#F87171"
                elif word_count > 800:
                    length_assessment = "Too Long (Consider condensing)"
                    length_color = "#FBBF24"
                else:
                    length_assessment = "Optimal Length"
                    length_color = "#34D399"
                st.markdown(f"**Length Assessment:** <span style='color: {length_color}; font-weight: bold'>{length_assessment}</span>", unsafe_allow_html=True)

                # Calculate readability
                try:
                    readability = textstat.flesch_reading_ease(resume_text)
                    if readability >= 60:
                        readability_text = "Easy to Read"
                        readability_color = "#34D399"
                    elif readability >= 50:
                        readability_text = "Fairly Easy to Read"
                        readability_color = "#FBBF24"
                    else:
                        readability_text = "Difficult to Read"
                        readability_color = "#F87171"
                    st.markdown(f"**Readability:** <span style='color: {readability_color}; font-weight: bold'>{readability_text}</span>", unsafe_allow_html=True)
                except:
                    pass

        with tab2:
            st.markdown('### 🎯 AI-Powered Role Predictions')
            if role_predictions.get('roles') and role_predictions.get('scores'):
                roles = role_predictions['roles']
                scores = role_predictions['scores']
                role_chart = create_role_prediction_chart(roles, scores)
                if role_chart:
                    st.plotly_chart(role_chart, use_container_width=True)
                st.markdown("#### 📋 Detailed Role Analysis")
                for i, (role, score) in enumerate(zip(roles[:5], scores[:5])):
                    with st.expander(f"{i+1}. {role} - {score:.1f}% match"):
                        st.markdown(f"**Career Path Recommendations:**")
                        st.markdown(f"- Senior {role} (3-5 years experience)")
                        st.markdown(f"- {role} Manager (Leadership path)")
                        st.markdown(f"- Related: {role.split()[0]} Specialist")
                        st.markdown(f"**Skill Development:**")
                        st.markdown(f"- Complete advanced certification in {role.split()[0]}")
                        st.markdown(f"- Develop leadership capabilities")
                        st.markdown(f"- Master industry-standard tools")
            else:
                st.info("No role predictions available. This might be due to missing ML models or insufficient resume data.")

        with tab3:
            if job_description.strip() and similarity_scores:
                from copy import deepcopy
                similarity_scores = deepcopy(similarity_scores)
                
                st.markdown("### 🎯 Job Description Match Analysis")
                
                col_metrics, col_gauge = st.columns([1, 1])
                
                with col_metrics:
                    tfidf_sim = similarity_scores.get('tfidf_similarity', 0.0)
                    keyword_sim = similarity_scores.get('keyword_similarity', 0.0)
                    combined_score = similarity_scores.get('combined_score', 0.0)
                    
                    st.metric("TF-IDF Similarity", f"{tfidf_sim:.1f}%", help="Algorithmic similarity based on overall document content")
                    st.metric("Keyword Similarity", f"{keyword_sim:.1f}%", help="Match based on important keywords in job description")
                    st.metric("Combined Score", f"{combined_score:.1f}%", help="Overall match score combining both algorithms")

                    similarity_data = pd.DataFrame({
                        'Metric': ['TF-IDF', 'Keywords', 'Combined'],
                        'Score': [tfidf_sim, keyword_sim, combined_score]
                    })

                    fig_bar = px.bar(
                        similarity_data, 
                        x='Score',
                        y='Metric',
                        orientation='h',
                        title="Similarity Breakdown",
                        color='Score',
                        color_continuous_scale='RdYlGn',
                        text='Score'
                    )
                    fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig_bar.update_layout(
                        height=300,
                        plot_bgcolor='#0f172a',
                        paper_bgcolor='#0f172a',
                        font={'color': '#f1f5f9'},
                        title_font={'size': 16, 'color': '#f1f5f9'},
                        xaxis_title="Match Score (%)",
                        yaxis_title="Metric",
                        xaxis=dict(color='#f1f5f9'),
                        yaxis=dict(color='#f1f5f9')
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with col_gauge:
                    # Added stylish box around the gauge
                    st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                        border: 1px solid #3b82f6;
                        border-radius: 12px;
                        padding: 20px;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    ">
                    """, unsafe_allow_html=True)
                    gauge_fig = create_match_gauge(combined_score)
                    st.plotly_chart(gauge_fig, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("#### 🔍 Matching Insights")
                combined_score = similarity_scores.get('combined_score', 0.0)
                
                # Determine match level and recommendations
                if combined_score >= 90:
                    match_level = "Perfect Match"
                    st.success(f"🎯 {match_level} ({combined_score:.1f}%)")
                    st.markdown("**Recommendations:**")
                    st.markdown("- Your resume perfectly matches the job requirements")
                    st.markdown("- Highlight your strongest qualifications at the top")
                    st.markdown("- Prepare specific examples of your achievements")
                elif combined_score >= 80:
                    match_level = "Excellent Match"
                    st.success(f"🌟 {match_level} ({combined_score:.1f}%)")
                    st.markdown("**Recommendations:**")
                    st.markdown("- Your resume shows excellent alignment with the job requirements")
                    st.markdown("- Quantify your achievements with specific metrics")
                    st.markdown("- Highlight your top 3 skills at the beginning")
                elif combined_score >= 70:
                    match_level = "Strong Match"
                    st.info(f"👍 {match_level} ({combined_score:.1f}%)")
                    st.markdown("**Recommendations:**")
                    st.markdown("- Your resume has strong alignment with the job requirements")
                    st.markdown("- Add 1-2 more keywords from the job description")
                    st.markdown("- Emphasize transferable skills")
                elif combined_score >= 60:
                    match_level = "Good Match"
                    st.warning(f"⚠️ {match_level} ({combined_score:.1f}%)")
                    st.markdown("**Recommendations:**")
                    st.markdown("- Your resume shows good alignment but could be improved")
                    st.markdown("- Add 3-5 more keywords from the job description")
                    st.markdown("- Include more metrics to quantify achievements")
                elif combined_score >= 40:
                    match_level = "Moderate Match"
                    st.warning(f"📝 {match_level} ({combined_score:.1f}%)")
                    st.markdown("**Recommendations:**")
                    st.markdown("- Your resume has moderate alignment with the job requirements")
                    st.markdown("- Focus on developing skills in gap areas")
                    st.markdown("- Consider taking relevant courses or certifications")
                else:
                    match_level = "Low Match"
                    st.error(f"❌ {match_level} ({combined_score:.1f}%)")
                    st.markdown("**Recommendations:**")
                    st.markdown("- Re-evaluate if this role aligns with your career goals")
                    st.markdown("- Consider similar but more junior positions")
                    st.markdown("- Focus on developing the required skills")

                missing_keywords = similarity_scores.get('missing_keywords', [])
                if missing_keywords:
                    st.markdown("#### ⚠️ Missing Keywords")
                    st.info("These important keywords from the job description were not found in your resume:")
                    cols = st.columns(3)
                    for i, keyword in enumerate(missing_keywords[:9]):
                        with cols[i % 3]:
                            st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #F87171 0%, #FB7185 100%);
                                        color: white; padding: 8px 12px; border-radius: 8px;
                                        margin: 5px; text-align: center; font-size: 0.9rem;'>
                                {keyword}
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.info("🔗 Provide a job description in the sidebar to see detailed matching analysis.")
                st.markdown("""
                #### Why Job Description Matching is Important:
                - **Precise Fit Assessment**: Compare candidate skills directly with job requirements
                - **Keyword Optimization**: Identify missing keywords for resume improvement
                - **Competitive Analysis**: Understand how well the resume matches specific roles
                - **Interview Preparation**: Focus on areas where skills align with job needs
                """)
        with tab4:
            st.markdown('### 💼 Personalized Job Recommendations')
            
            if analysis_results.get('job_suggestions'):
                job_suggestions = analysis_results['job_suggestions']
                
                for i, job in enumerate(job_suggestions[:5]):  # Show top 5 suggestions
                    # Added hover effect and border to job cards
                    st.markdown(f"""
                    <div class="job-card" style="
                        border-radius: 12px;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                        padding: 20px;
                        margin-bottom: 20px;
                        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                        border: 1px solid #334155;
                    ">
                        <h4 style="margin-top:0; margin-bottom:10px; color:#f1f5f9;">
                            🎯 {job.get('role', 'N/A')}
                            <span style="float: right; background: rgba(59, 130, 246, 0.2);
                                         padding: 5px 15px; border-radius: 20px; font-size: 0.85rem; color:#3b82f6;">
                                Match: {job.get('match_score', 0):.1f}%
                            </span>
                        </h4>
                        <p><strong>🏢 Company:</strong> {job.get('company', 'Leading Tech Company')}</p>
                        <p><strong>📍 Location:</strong> {job.get('location', 'Remote')}</p>
                        <p><strong>💰 Salary Range:</strong> {job.get('salary_range', '$90K - $140K')}</p>
                        <p><strong>🛠 Key Skills:</strong> {', '.join(job.get('required_skills', ['Python', 'ML', 'Data Analysis'])[:5])}</p>
                        <p><strong>💡 Recommendation:</strong> {job.get('recommendation_reason', 'Your skills align well with this position')}</p>
                        <button style="
                            background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
                            border: none;
                            color: white;
                            padding: 8px 20px;
                            border-radius: 8px;
                            margin-top: 10px;
                            cursor: pointer;
                            font-weight: 600;
                        ">Apply Now</button>
                    </div>
                    """, unsafe_allow_html=True)
                    
                if len(job_suggestions) > 1:
                    job_data = pd.DataFrame({
                        'Role': [job.get('role', 'N/A') for job in job_suggestions[:5]],
                        'Match Score': [job.get('match_score', 0) for job in job_suggestions[:5]]
                    })

                    fig_jobs = px.bar(
                        job_data,
                        x='Match Score',
                        y='Role',
                        orientation='h',
                        title="Job Recommendation Scores",
                        color='Match Score',
                        color_continuous_scale='Viridis',
                        text='Match Score'
                    )
                    fig_jobs.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig_jobs.update_layout(
                        height=400,
                        plot_bgcolor='#0f172a',
                        paper_bgcolor='#0f172a',
                        font={'color': '#f1f5f9'},
                        title_font={'size': 18, 'color': '#f1f5f9'},
                        xaxis_title="Match Score (%)",
                        yaxis_title="Job Role",
                        xaxis=dict(color='#f1f5f9'),
                        yaxis=dict(color='#f1f5f9')
                    )
                    st.plotly_chart(fig_jobs, use_container_width=True)
            else:
                st.info("No job recommendations generated. This might be due to insufficient data or missing ML models.")
                st.markdown("""
                #### 💡 Tips for Better Job Recommendations:
                - Ensure your resume contains clear skill descriptions
                - Include relevant work experience details
                - Add education and certification information
                - Use industry-standard terminology and keywords
                """)

        st.markdown("---")
        st.markdown('### 📋 Analysis Summary')

        if analysis_results.get("analysis_summary"):
            st.success(f"✅ {analysis_results['analysis_summary']}")
        else:
            st.info("✅ Resume analysis completed successfully! Review the detailed insights above.")

        st.markdown("#### 🎯 Recommended Next Steps:")
        action_items = []
        skills_to_learn = []

        # Basic resume completeness checks
        if parsed_info.get('name') in [None, 'Not Mentioned']:
            action_items.append("Add a clear name/header to your resume")
        if parsed_info.get('email') in [None, 'Not Mentioned']:
            action_items.append("Include a professional email address")
        if len(parsed_info.get('skills', [])) < 5:
            action_items.append("Add more relevant technical and soft skills")

        # JD-specific recommendations
        if job_description.strip() and similarity_scores:
            # Always show missing keywords from JD if available
            if similarity_scores.get('missing_keywords'):
                skills_to_learn = similarity_scores['missing_keywords']
                action_items.append(f"Add these skills to better match the job description: {', '.join(skills_to_learn[:5])}...")

            # Add score-specific recommendations
            combined_score = similarity_scores.get('combined_score', 0)
            if combined_score < 60:
                action_items.append("Quantify achievements with specific metrics")
                action_items.append("Emphasize transferable skills")

        if action_items:
            for item in action_items:
                st.markdown(f"""
                <div style="
                    background: rgba(59, 130, 246, 0.2);
                    padding: 12px;
                    border-radius: 12px;
                    margin: 10px 0;
                    color: #f1f5f9;
                    font-weight: 600;
                ">⚠️ {item}</div>
                """, unsafe_allow_html=True)
        else:
            st.success("🎉 Your resume looks comprehensive! Consider tailoring it for specific job applications.")
            
            # Show JD skills even when resume is comprehensive
            if job_description.strip() and similarity_scores and similarity_scores.get('missing_keywords'):
                skills_to_learn = similarity_scores['missing_keywords']
                st.markdown("#### 💡 Skills to Highlight for This Role")
                st.info("These skills from the job description could be emphasized more in your resume:")
                cols = st.columns(3)
                for i, skill in enumerate(skills_to_learn[:9]):
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
                                    color: white; padding: 8px 12px; border-radius: 8px;
                                    margin: 5px; text-align: center; font-size: 0.9rem;'>
                            {skill}
                        </div>
                        """, unsafe_allow_html=True)

        # Generate and download report
        st.markdown("---")
        st.markdown("### 📊 Download Resources")
        
        # Create attractive report
        report_html = create_attractive_report(
            parsed_info, 
            role_predictions, 
            similarity_scores if job_description.strip() else None,
            action_items, 
            skills_to_learn,
            job_title
        )
        
        # Create ATS-friendly resume
        resume_pdf = create_ats_friendly_pdf(
            parsed_info,
            skills_to_learn,
            job_title
        )
        
        # Download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(create_download_link(
                report_html, 
                "resume_analysis_report.html", 
                "📊 Download Analysis Report"
            ), unsafe_allow_html=True)
            st.info("Comprehensive analysis with visual insights")
        
        with col2:
            st.markdown(create_pdf_download_link(
                resume_pdf, 
                "ats_friendly_resume.pdf", 
                "📝 Download ATS Resume"
            ), unsafe_allow_html=True)
            st.info("Optimized resume template with your information")

        st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.15); padding: 20px; border-radius: 12px; margin-top: 20px;">
            <h3>🚀 Resume Optimization Tips</h3>
            <p>Our ATS-friendly resume includes these key optimizations:</p>
            <ul>
                <li><strong>Professional Header</strong> with clear contact information</li>
                <li><strong>ATS-Optimized Format</strong> - Passes through all major applicant tracking systems</li>
                <li><strong>Keyword Optimization</strong> - Includes recommended skills from job description</li>
                <li><strong>Quantified Achievements</strong> - Demonstrates impact with numbers and metrics</li>
                <li><strong>Clean, Professional Formatting</strong> - No columns or complex layouts</li>
                <li><strong>Standard Section Headings</strong> - Easily parsable by ATS systems</li>
                <li><strong>Relevant Certifications</strong> - Added to increase credibility</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

elif uploaded_file and not analyze_button:
    st.info("👆 Click the 'Start Deep Analysis' button in the sidebar to process your resume.")
    
    # Preview section
    st.markdown("""
    <div style="max-width: 1200px; margin: 40px auto;">
      <div style="display: flex; gap: 20px; flex-wrap: wrap; justify-content: center;">
        <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
                    padding: 20px; border-radius: 12px; width: 300px; 
                    border: 1px solid #334155; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h4 style="color: #f1f5f9;">📄 Resume Analysis</h4>
            <ul style="color:#94a3b8; font-size: 1rem;">
                <li>Personal information extraction</li>
                <li>Skills identification</li>
                <li>Visual skills analysis</li>
                <li>Resume completeness</li>
                <li>Text quality metrics</li>
            </ul>
        </div>
        <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
                    padding: 20px; border-radius: 12px; width: 300px; 
                    border: 1px solid #334155; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h4 style="color: #f1f5f9;">🎯 AI Predictions</h4>
            <ul style="color:#94a3b8; font-size: 1rem;">
                <li>Job role recommendations</li>
                <li>Career path suggestions</li>
                <li>Confidence scoring</li>
                <li>Industry alignment</li>
                <li>Growth opportunities</li>
            </ul>
        </div>
        <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
                    padding: 20px; border-radius: 12px; width: 300px; 
                    border: 1px solid #334155; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h4 style="color: #f1f5f9;">🔗 Job Matching</h4>
            <ul style="color:#94a3b8; font-size: 1rem;">
                <li>JD similarity scoring</li>
                <li>Keyword optimization</li>
                <li>Competitive analysis</li>
                <li>Interview preparation</li>
                <li>Personalized feedback</li>
            </ul>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # Welcome Screen
    st.markdown("""
    <div style="max-width: 900px; margin: 60px auto; 
                background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
                border-radius: 12px; padding: 40px; 
                border: 1px solid #334155; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        <h1 class="main-title">👋 Welcome to Resume Analyzer Pro!</h1>
        <p style="color: #94a3b8; font-size: 18px; text-align: center; margin-top: 20px;">
          Upload your resume to get started with our comprehensive AI-powered analysis. Our system will provide detailed insights including:
        </p>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 24px; margin-top: 30px;">
            <div style="background: rgba(30, 41, 59, 0.5); padding: 20px; border-radius: 12px; 
                        border: 1px solid #334155; color: #f1f5f9;">
                🔍 <strong>Deep Resume Analysis</strong> - Extract and analyze all resume components
            </div>
            <div style="background: rgba(30, 41, 59, 0.5); padding: 20px; border-radius: 12px; 
                        border: 1px solid #334155; color: #f1f5f9;">
                🎯 <strong>AI Role Predictions</strong> - Discover best-fitting job roles
            </div>
            <div style="background: rgba(30, 41, 59, 0.5); padding: 20px; border-radius: 12px; 
                        border: 1px solid #334155; color: #f1f5f9;">
                🔗 <strong>Job Matching</strong> - Compare against specific job descriptions
            </div>
            <div style="background: rgba(30, 41, 59, 0.5); padding: 20px; border-radius: 12px; 
                        border: 1px solid #334155; color: #f1f5f9;">
                💼 <strong>Recommendations</strong> - Get tailored career advice
            </div>
            <div style="background: rgba(30, 41, 59, 0.5); padding: 20px; border-radius: 12px; 
                        border: 1px solid #334155; color: #f1f5f9;">
                📊 <strong>Visual Analytics</strong> - Interactive charts and insights
            </div>
            <div style="background: rgba(30, 41, 59, 0.5); padding: 20px; border-radius: 12px; 
                        border: 1px solid #334155; color: #f1f5f9;">
                ⚡ <strong>Optimization Tips</strong> - Actionable improvement suggestions
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.info("📁 Use the sidebar to upload your resume and start the analysis!")

# Footer
st.markdown("""
<div style="max-width: 1200px; margin: 40px auto; text-align: center; color: #94a3b8; font-size: 14px; padding: 20px;">
    🚀 Resume Analyzer Pro - Powered by Advanced Machine Learning | v2.1 <br/>
    💡 AI-driven insights for career advancement | Always verify with human judgment
</div>
""", unsafe_allow_html=True) 
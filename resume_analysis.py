# ABSOLUTE FIRST STREAMLIT COMMAND
import streamlit as st
st.set_page_config(
    page_title="Resume Analyzer Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# THEN ALL OTHER IMPORTS
import os
import re
import json
import warnings
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Traditional ML imports only (no NLP)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Rest of your code continues...
warnings.filterwarnings("ignore")

# HELPER FUNCTION DEFINITIONS
class JobMatchingSystem:
    def __init__(self):
        print("CONSOLE: Initializing ML-only Job Matching System...")
        
        # Load ML models
        try:
            self.clf_model = self._load_model('clf.pkl')
            print("CONSOLE: Classifier model loaded")
        except:
            print("CONSOLE: Failed to load classifier model")
            self.clf_model = None
            
        try:
            self.tfidf = self._load_model('tfidf.pkl')
            print("CONSOLE: TF-IDF vectorizer loaded")
        except:
            print("CONSOLE: Failed to load TF-IDF vectorizer")
            self.tfidf = None
            
        try:
            self.encoder = self._load_model('encoder.pkl')
            print("CONSOLE: Label encoder loaded")
        except:
            print("CONSOLE: Failed to label encoder")
            self.encoder = None
        
        # Job roles and skills database
        self.job_roles = [
            "Data Analyst", "Software Engineer", "Frontend Developer", 
            "Backend Developer", "Full Stack Developer", "Machine Learning Engineer", 
            "DevOps Engineer", "Business Analyst", "Product Manager", 
            "UI/UX Designer", "Data Scientist", "Cybersecurity Analyst",
            "Mobile Developer", "Cloud Engineer", "System Administrator"
        ]
        
        self.skills_database = {
            "Programming": ["Python", "Java", "JavaScript", "C++", "C#", "R", "Go", "Rust", "PHP", "Ruby"],
            "Web Development": ["HTML", "CSS", "React", "Angular", "Vue.js", "Node.js", "Express", "Django", "Flask"],
            "Data Science": ["Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch", "Matplotlib", "Seaborn"],
            "Databases": ["SQL", "MySQL", "PostgreSQL", "MongoDB", "Redis", "Cassandra", "Oracle"],
            "Cloud": ["AWS", "Azure", "Google Cloud", "Docker", "Kubernetes", "Terraform"],
            "Tools": ["Git", "JIRA", "Slack", "Tableau", "Power BI", "Excel", "Jupyter"]
        }
        
        print("CONSOLE: Job Matching System initialized successfully!")

    def _load_model(self, filename):
        """Load ML model from pickle file"""
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def clean_resume(self, text):
        """Clean resume text (simplified version)"""
        text = re.sub(r'http\S+\s', ' ', text)  # Remove URLs
        text = re.sub(r'@\S+', ' ', text)       # Remove mentions
        text = re.sub(r'#\S+\s', ' ', text)      # Remove hashtags
        text = re.sub(r'\s+', ' ', text)         # Remove extra whitespace
        return text.lower().strip()

    def parse_resume(self, resume_text: str) -> Dict:
        """Resume parsing using regex only"""
        parsed_data = {}
        parsed_data['email'] = self._extract_email(resume_text)
        parsed_data['phone'] = self._extract_phone(resume_text)
        parsed_data['name'] = self._extract_name(resume_text)
        parsed_data['skills'] = self._extract_skills(resume_text)
        parsed_data['experience'] = self._extract_experience(resume_text)
        parsed_data['education'] = self._extract_education(resume_text)
        return parsed_data

    def _extract_email(self, text: str) -> str:
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        return emails[0] if emails else "Not found"

    def _extract_phone(self, text: str) -> str:
        phone_patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            r'\(\d{3}\)\s*\d{3}[-.]?\d{4}',
            r'\+\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{4}'
        ]
        for pattern in phone_patterns:
            phones = re.findall(pattern, text)
            if phones:
                return phones[0]
        return "Not found"

    def _extract_name(self, text: str) -> str:
        lines = text.strip().split('\n')
        for line in lines[:5]:
            line = line.strip()
            if line and len(line.split()) <= 4 and not '@' in line and not any(char.isdigit() for char in line):
                return line
        return "Not found"

    def _extract_skills(self, text: str) -> List[str]:
        skills_found = []
        text_lower = text.lower()
        for category, skills_list in self.skills_database.items():
            for skill in skills_list:
                if skill.lower() in text_lower:
                    skills_found.append(skill)
        return list(set(skills_found))

    def _extract_experience(self, text: str) -> str:
        experience_patterns = [
            r'(\d+)[\+\-\s]years?[\s](?:of\s*)?experience',
            r'experience[\s]:?[\s](\d+)[\+\-\s]*years?',
            r'(\d+)[\s](?:to\s\d+)?[\s]years?[\s](?:of\s*)?(?:work\s*)?experience'
        ]
        for pattern in experience_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                return f"{matches[0]} years"
        return "Not specified"

    def _extract_education(self, text: str) -> List[str]:
        education_keywords = [
            "bachelor", "master", "phd", "degree", "university", "college",
            "b.tech", "m.tech", "mba", "bca", "mca", "b.sc", "m.sc"
        ]
        education_found = []
        text_lower = text.lower()
        for keyword in education_keywords:
            if keyword in text_lower:
                lines = text.split('\n')
                for line in lines:
                    if keyword in line.lower():
                        education_found.append(line.strip())
                        break
        return list(set(education_found)) if education_found else ["Not specified"]

    def calculate_similarity_scores(self, resume_text: str, job_description: str) -> Dict:
        scores = {}
        
        # TF-IDF Cosine Similarity
        scores['tfidf_similarity'] = self._tfidf_similarity(resume_text, job_description)
        
        # Keyword matching score
        scores['keyword_similarity'] = self._keyword_similarity(resume_text, job_description)
        
        # Combined score
        scores['combined_score'] = (
            scores['tfidf_similarity'] * 0.6 + 
            scores['keyword_similarity'] * 0.4
        )
        
        return scores

    def _tfidf_similarity(self, text1: str, text2: str) -> float:
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return round(similarity * 100, 2)
        except:
            return 0.0

    def _keyword_similarity(self, resume_text: str, job_description: str) -> float:
        resume_keywords = set(re.findall(r'\b\w+\b', resume_text.lower()))
        job_keywords = set(re.findall(r'\b\w+\b', job_description.lower()))
        common_keywords = resume_keywords.intersection(job_keywords)
        total_keywords = resume_keywords.union(job_keywords)
        return round((len(common_keywords) / len(total_keywords)) * 100, 2) if total_keywords else 0.0

    def predict_job_roles(self, resume_text: str) -> Dict:
        """Predict job roles using ML classifier"""
        if self.clf_model and self.tfidf and self.encoder:
            try:
                cleaned_text = self.clean_resume(resume_text)
                vectorized_text = self.tfidf.transform([cleaned_text])
                prediction = self.clf_model.predict(vectorized_text)
                probabilities = self.clf_model.predict_proba(vectorized_text)[0]
                
                # Get top 5 predictions
                top_indices = np.argsort(probabilities)[::-1][:5]
                roles = self.encoder.inverse_transform(top_indices)
                scores = [round(probabilities[i] * 100, 2) for i in top_indices]
                
                return {"roles": roles.tolist(), "scores": scores}
            except Exception as e:
                print(f"CONSOLE: Prediction error: {e}")
        
        # Fallback if ML models not available
        return self._fallback_role_prediction(resume_text)

    def _fallback_role_prediction(self, resume_text: str) -> Dict:
        """Fallback role prediction using keyword matching"""
        role_keywords = {
            "Data Analyst": ["data", "analyst", "sql", "excel", "tableau"],
            "Software Engineer": ["software", "engineer", "programming"],
            "Frontend Developer": ["frontend", "html", "css", "javascript"],
            "Backend Developer": ["backend", "api", "server", "database"],
            "Machine Learning Engineer": ["machine learning", "ml", "tensorflow"],
            "DevOps Engineer": ["devops", "docker", "kubernetes"],
            "Data Scientist": ["data science", "statistics", "python"]
        }
        text_lower = resume_text.lower()
        role_scores = {role: 0 for role in role_keywords}
        
        for role, keywords in role_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    role_scores[role] += 1
        
        # Sort by score
        sorted_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)
        roles = [role for role, _ in sorted_roles[:5]]
        scores = [min(score * 20, 100) for _, score in sorted_roles[:5]]  # Scale to 0-100
        
        return {"roles": roles, "scores": scores}

    def generate_job_suggestions(self, resume_data: Dict, preferences: Dict = None) -> List[Dict]:
        suggestions = []
        role_prediction = self.predict_job_roles(' '.join(str(v) for v in resume_data.values()))
        
        for i, (role, score) in enumerate(zip(role_prediction["roles"][:5], 
                                            role_prediction["scores"][:5])):
            suggestion = {
                "role": role,
                "match_score": score,
                "required_skills": self._get_role_requirements(role),
                "salary_range": self._estimate_salary_range(role, resume_data.get('experience', '0')),
                "recommendation_reason": f"Based on your skills in {', '.join(resume_data.get('skills', ['various technologies'])[:3])}"
            }
            suggestions.append(suggestion)
        
        return suggestions

    def _get_role_requirements(self, role: str) -> List[str]:
        role_requirements = {
            "Data Analyst": ["SQL", "Excel", "Python", "Tableau"],
            "Software Engineer": ["Programming", "Algorithms", "Git"],
            "Frontend Developer": ["HTML", "CSS", "JavaScript"],
            "Backend Developer": ["Python/Java", "Databases", "APIs"],
            "Machine Learning Engineer": ["Python", "ML Algorithms"],
            "DevOps Engineer": ["Docker", "Kubernetes", "Cloud"],
            "Data Scientist": ["Python", "Statistics", "ML"]
        }
        return role_requirements.get(role, ["Problem Solving", "Communication"])

    def _estimate_salary_range(self, role: str, experience: str) -> str:
        base_salaries = {
            "Data Analyst": 60000,
            "Software Engineer": 80000,
            "Frontend Developer": 75000,
            "Backend Developer": 85000,
            "Machine Learning Engineer": 95000,
            "DevOps Engineer": 90000,
            "Data Scientist": 100000
        }
        base = base_salaries.get(role, 70000)
        exp_years = 0
        if isinstance(experience, str):
            exp_match = re.search(r'(\d+)', experience)
            if exp_match:
                exp_years = int(exp_match.group(1))
        adjusted_salary = base + (exp_years * 5000)
        return f"${adjusted_salary:,} - ${adjusted_salary + 20000:,}"

    def analyze_resume_complete(self, resume_text: str, job_description: str = None, 
                              preferences: Dict = None) -> Dict:
        parsed_data = self.parse_resume(resume_text)
        similarity_scores = {}
        if job_description:
            similarity_scores = self.calculate_similarity_scores(resume_text, job_description)
        
        role_predictions = self.predict_job_roles(resume_text)
        job_suggestions = self.generate_job_suggestions(parsed_data, preferences)
        
        complete_analysis = {
            "timestamp": datetime.now().isoformat(),
            "parsed_resume": parsed_data,
            "similarity_scores": similarity_scores,
            "role_predictions": role_predictions,
            "job_suggestions": job_suggestions,
            "analysis_summary": f"Analysis completed for {parsed_data.get('name', 'candidate')}"
        }
        
        return complete_analysis

# Streamlit UI
print("CONSOLE: Starting Streamlit app...")

@st.cache_resource
def load_jms_cached():
    print("CONSOLE: Loading JobMatchingSystem...")
    return JobMatchingSystem()

jms_system = load_jms_cached()

st.markdown("""
<style>
    .stButton>button { font-size: 1.1rem; padding: 0.6rem 1.2rem; }
</style>
""", unsafe_allow_html=True)

st.title("📄 Resume Analyzer Pro (ML Edition)")
st.caption(f"Machine Learning Version | {datetime.now().strftime('%B %d, %Y')}")

with st.sidebar:
    st.subheader("🔍 Model Status")
    if jms_system.clf_model:
        st.success("✓ ML Models Loaded")
    else:
        st.warning("⚠ Some models not loaded")
    st.info("This version uses traditional ML only")

uploaded_file = st.file_uploader("Upload your Resume", type=["pdf", "docx", "txt"])
job_description = st.text_area("Job Description (Optional):", height=150)

if uploaded_file:
    resume_text = ""
    try:
        if uploaded_file.name.endswith('.pdf'):
            # Simplified PDF extraction
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                resume_text += page.extract_text() or ''
        elif uploaded_file.name.endswith('.docx'):
            # Simplified DOCX extraction
            import docx
            doc = docx.Document(uploaded_file)
            resume_text = "\n".join([para.text for para in doc.paragraphs])
        else:
            resume_text = uploaded_file.getvalue().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
    
    if st.button("🚀 Analyze Resume", type="primary", use_container_width=True):
        with st.spinner("Analyzing with ML models..."):
            analysis_results = jms_system.analyze_resume_complete(
                resume_text, 
                job_description=job_description if job_description else None
            )
        
        st.balloons()
        st.header("📊 Analysis Results")
        
        # Display parsed resume data
        with st.expander("📝 Parsed Resume Data"):
            st.json(analysis_results['parsed_resume'])
        
        # Display similarity scores
        if job_description:
            st.subheader("🔍 Similarity Scores")
            cols = st.columns(3)
            cols[0].metric("TF-IDF Similarity", f"{analysis_results['similarity_scores'].get('tfidf_similarity', 0)}%")
            cols[1].metric("Keyword Similarity", f"{analysis_results['similarity_scores'].get('keyword_similarity', 0)}%")
            cols[2].metric("Overall Match", f"{analysis_results['similarity_scores'].get('combined_score', 0)}%", delta="Job Fit")
        
        # Display job role predictions
        st.subheader("🎯 Predicted Job Roles")
        roles = analysis_results['role_predictions']['roles']
        scores = analysis_results['role_predictions']['scores']
        for role, score in zip(roles[:3], scores[:3]):
            st.progress(int(score), text=f"{role} ({score}%)")
        
        # Display job suggestions
        st.subheader("💼 Recommended Jobs")
        for job in analysis_results['job_suggestions'][:3]:
            with st.expander(f"{job['role']} - Match: {job['match_score']}%"):
                st.markdown(f"**Salary Range:** {job['salary_range']}")
                st.markdown(f"**Key Skills Needed:** {', '.join(job['required_skills'][:5])}")
                st.markdown(f"**Why this role:** {job['recommendation_reason']}")
        
        st.success("✅ Analysis completed successfully!")

print("CONSOLE: End of Streamlit script reached.")
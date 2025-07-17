# --- contents of dummy.py ---
import re
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer # Assuming these were used for training
from sklearn.metrics.pairwise import cosine_similarity    # and saved in pkl

# Traditional ML imports only (no NLP beyond TF-IDF for features)
# from sklearn.cluster import KMeans # Not used in active path
# from sklearn.preprocessing import LabelEncoder # Loaded from pickle
# from sklearn.ensemble import RandomForestClassifier # Loaded from pickle
# from sklearn.svm import SVC # Loaded from pickle


class JobMatchingSystem:
    def __init__(self):
        print("CONSOLE: Initializing ML-only Job Matching System...")
        
        # Load ML models
        try:
            # Ensure these .pkl files exist in the same directory or provide full paths
            self.clf_model = self._load_model('clf.pkl')
            print("CONSOLE: Classifier model loaded")
        except FileNotFoundError:
            print("CONSOLE: Failed to load classifier model (clf.pkl not found). Fallback will be used.")
            self.clf_model = None
        except Exception as e:
            print(f"CONSOLE: Error loading classifier model: {e}")
            self.clf_model = None
            
        try:
            self.tfidf = self._load_model('tfidf.pkl')
            print("CONSOLE: TF-IDF vectorizer loaded")
        except FileNotFoundError:
            print("CONSOLE: Failed to load TF-IDF vectorizer (tfidf.pkl not found).")
            self.tfidf = None
        except Exception as e:
            print(f"CONSOLE: Error loading TF-IDF vectorizer: {e}")
            self.tfidf = None
            
        try:
            self.encoder = self._load_model('encoder.pkl')
            print("CONSOLE: Label encoder loaded")
        except FileNotFoundError:
            print("CONSOLE: Failed to load label encoder (encoder.pkl not found).")
            self.encoder = None
        except Exception as e:
            print(f"CONSOLE: Error loading label encoder: {e}")
            self.encoder = None
        
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
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def clean_resume(self, text):
        text = re.sub(r'http\S+\s', ' ', text)
        text = re.sub(r'@\S+', ' ', text)
        text = re.sub(r'#\S+\s', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()

    def parse_resume(self, resume_text: str) -> Dict:
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
        for line in lines[:5]: # Check first 5 lines for a name
            line = line.strip()
            # Basic check: not an email, no digits, plausible length
            if line and len(line.split()) <= 4 and '@' not in line and not any(char.isdigit() for char in line) and len(line) > 3:
                # Avoid lines that look like section headers (e.g., "SKILLS", "EDUCATION")
                if line.upper() not in ["SKILLS", "EDUCATION", "EXPERIENCE", "PROJECTS", "SUMMARY", "OBJECTIVE", "CONTACT"]:
                    return line
        return "Not found"


    def _extract_skills(self, text: str) -> List[str]:
        skills_found = []
        text_lower = text.lower()
        # Consider adding a regex for skills if they follow a pattern, or look in a specific "Skills" section
        for category, skills_list in self.skills_database.items():
            for skill in skills_list:
                # Use word boundaries to avoid partial matches e.g. 'C' in 'Certificate'
                if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
                    skills_found.append(skill)
        return list(set(skills_found))

    def _extract_experience(self, text: str) -> str:
        experience_patterns = [
            r'(\d+\.?\d*|\d+)[\+\-\s]years?[\s](?:of\s*)?experience', # "5+ years experience", "3 years of experience"
            r'experience[\s:]?[\s]*(\d+\.?\d*|\d+)[\+\-\s]*years?',   # "experience: 5 years"
            r'(\d+\.?\d*|\d+)[\s](?:to\s\d+\.?\d*|\d+)?[\s]years?[\s](?:of\s*)?(?:work\s*)?experience' # "3 to 5 years experience"
        ]
        text_to_search = text.lower()
        # Prioritize sections typically containing overall experience
        experience_section_keywords = ['summary', 'objective', 'overview', 'profile']
        search_priority_text = ""
        
        for keyword in experience_section_keywords:
            match = re.search(rf"{keyword}.*?\n\n", text_to_search, re.IGNORECASE | re.DOTALL) # crude section end
            if match:
                search_priority_text += match.group(0) + " "

        if not search_priority_text: # if no specific sections found, search whole text
            search_priority_text = text_to_search

        for pattern in experience_patterns:
            matches = re.findall(pattern, search_priority_text)
            if matches:
                # If multiple numbers are captured due to groups in regex, try to find the most relevant one.
                # For now, taking the first plausible number.
                for match_tuple in matches:
                    if isinstance(match_tuple, tuple):
                        for item in match_tuple:
                            if item and item.replace('.', '', 1).isdigit(): return f"{item} years"
                    elif isinstance(match_tuple, str) and match_tuple.replace('.', '', 1).isdigit():
                        return f"{match_tuple} years"
        
        # Fallback: if no pattern match, look for any "X years" text near 'experience'
        # This is very basic and may not be accurate
        simple_exp_match = re.search(r'(\d+)\s*year[s]?', text_to_search)
        if simple_exp_match:
            return f"{simple_exp_match.group(1)} years"
            
        return "Not specified"

    def _extract_education(self, text: str) -> List[str]:
        education_found = []
        text_lower = text.lower()
        
        # Look for an "Education" section first
        education_section = ""
        education_header_match = re.search(r'(education\b.*?)(?=\n\s*[A-Z][a-z]+[A-Z\s]*:|\n\n|\Z)', text_lower, re.IGNORECASE | re.DOTALL)
        if education_header_match:
            education_section = education_header_match.group(1)
        else: # if no explicit section, search the whole text (less reliable)
            education_section = text_lower

        education_keywords_degree = [
            r"bachelor(?:'s)?\s*(?:degree)?\s*(?:of|in|from)?\s*[\w\s]+", 
            r"master(?:'s)?\s*(?:degree)?\s*(?:of|in|from)?\s*[\w\s]+",
            r"phd\s*(?:in|from)?\s*[\w\s]+", 
            r"associate(?:'s)?\s*(?:degree)?\s*(?:of|in|from)?\s*[\w\s]+",
            r"m\.?b\.?a\.?", r"b\.?s\.?c\.?", r"m\.?s\.?c\.?", 
            r"b\.?tech\.?", r"m\.?tech\.?", 
            r"b\.?e\.?", # Bachelor of Engineering
            r"m\.?e\.?", # Master of Engineering
            r"college\s*(?:degree|diploma)?\s*(?:in|of|from)?\s*[\w\s]+",
            r"university\s*(?:degree|diploma)?\s*(?:in|of|from)?\s*[\w\s]+"
        ]
        
        # Split the education section by lines to find relevant degrees
        lines_to_search = [line.strip() for line in education_section.split('\n') if line.strip()]

        for line in lines_to_search:
            for keyword_pattern in education_keywords_degree:
                match = re.search(keyword_pattern, line, re.IGNORECASE)
                if match:
                    # Attempt to extract a meaningful segment around the keyword
                    context_window = 50 # characters before and after
                    start_index = max(0, match.start() - context_window)
                    end_index = min(len(line), match.end() + context_window)
                    # Try to get a more "complete" line for context
                    original_line_match = [orig_line for orig_line in text.split('\n') if line in orig_line.lower().strip()]
                    if original_line_match:
                         education_found.append(original_line_match[0].strip())
                    else:
                         education_found.append(line[start_index:end_index].strip())
                    break # Move to next line once a keyword is found in current line
        
        if not education_found:
            # Broader search if specific patterns fail - less accurate
            generic_keywords = ["degree", "university", "college", "diploma", "institute"]
            for line in lines_to_search:
                for gk in generic_keywords:
                    if gk in line:
                        original_line_match = [orig_line for orig_line in text.split('\n') if line in orig_line.lower().strip()]
                        if original_line_match:
                             education_found.append(original_line_match[0].strip())
                        else:
                             education_found.append(line.strip())
                        break # one find per line is enough for generic
                if len(education_found) > 3: # limit generic finds
                    break

        return list(set(education_found)) if education_found else ["Not specified"]


    def calculate_similarity_scores(self, resume_text: str, job_description: str) -> Dict:
        scores = {}
        if not resume_text or not job_description: # Basic check
             scores['tfidf_similarity'] = 0.0
             scores['keyword_similarity'] = 0.0
             scores['combined_score'] = 0.0
             return scores

        scores['tfidf_similarity'] = self._tfidf_similarity(resume_text, job_description)
        scores['keyword_similarity'] = self._keyword_similarity(resume_text, job_description)
        scores['combined_score'] = round((
            scores['tfidf_similarity'] * 0.6 + 
            scores['keyword_similarity'] * 0.4
        ), 2)
        return scores

    def _tfidf_similarity(self, text1: str, text2: str) -> float:
        if not self.tfidf: # Check if TF-IDF model (vectorizer) is loaded
            print("CONSOLE: TF-IDF vectorizer not loaded, cannot calculate TF-IDF similarity.")
             # Fallback or re-initialize if appropriate and possible, here just return 0
            try:
                print("CONSOLE: Attempting to initialize a default TF-IDF for this similarity calculation.")
                local_tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
                local_tfidf_vectorizer.fit([text1, text2]) # Fit on current texts
                tfidf_matrix = local_tfidf_vectorizer.transform([text1, text2])
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                return round(similarity * 100, 2)
            except Exception as e:
                print(f"CONSOLE: Error initializing default TF-IDF for similarity: {e}")
                return 0.0

        try:
            # Use the pre-loaded TF-IDF (which should have been fit on a larger corpus)
            # For direct comparison, it's often better to fit_transform on the combined texts
            # or ensure the existing self.tfidf is appropriate.
            # If self.tfidf was trained on a general corpus, transform both texts with it.
            # If it was trained on job descriptions only, its applicability to resume text for similarity might vary.
            # A common approach for comparing two specific docs:
            temp_vectorizer = TfidfVectorizer(vocabulary=self.tfidf.vocabulary_, stop_words='english') # Use existing vocab
            # Or, more simply, just use a new vectorizer for this pair for pairwise similarity
            # vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
            # tfidf_matrix = vectorizer.fit_transform([text1, text2])

            tfidf_matrix = self.tfidf.transform([self.clean_resume(text1), self.clean_resume(text2)]) # Use loaded and globally trained TFIDF
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return round(similarity * 100, 2)
        except Exception as e:
            print(f"CONSOLE: TF-IDF similarity calculation error: {e}")
            # Fallback to new vectorizer for this pair
            try:
                vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
                tfidf_matrix = vectorizer.fit_transform([self.clean_resume(text1), self.clean_resume(text2)])
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                return round(similarity * 100, 2)
            except:
                 return 0.0


    def _keyword_similarity(self, resume_text: str, job_description: str) -> float:
        try:
            resume_keywords = set(re.findall(r'\b\w+\b', self.clean_resume(resume_text).lower()))
            job_keywords = set(re.findall(r'\b\w+\b', self.clean_resume(job_description).lower()))
            
            if not resume_keywords or not job_keywords: # Handle empty sets
                return 0.0

            common_keywords = resume_keywords.intersection(job_keywords)
            # Optional: weight common keywords or consider only relevant ones
            
            # Jaccard Index variant
            total_unique_keywords = resume_keywords.union(job_keywords)
            if not total_unique_keywords: return 0.0
            
            return round((len(common_keywords) / len(total_unique_keywords)) * 100, 2)
        except Exception as e:
            print(f"CONSOLE: Keyword similarity calculation error: {e}")
            return 0.0

    def predict_job_roles(self, resume_text: str) -> Dict:
        if self.clf_model and self.tfidf and self.encoder:
            try:
                cleaned_text = self.clean_resume(resume_text)
                # Ensure tfidf is a TfidfVectorizer instance and has been fitted
                if not hasattr(self.tfidf, 'transform'):
                    print("CONSOLE: TF-IDF model is not a fitted vectorizer.")
                    return self._fallback_role_prediction(resume_text)
                
                vectorized_text = self.tfidf.transform([cleaned_text])
                
                # Ensure clf_model supports predict_proba and predict
                if not (hasattr(self.clf_model, 'predict_proba') and hasattr(self.clf_model, 'predict')):
                     print("CONSOLE: Classifier model does not support predict/predict_proba.")
                     return self._fallback_role_prediction(resume_text)

                prediction = self.clf_model.predict(vectorized_text) # Integer labels
                probabilities = self.clf_model.predict_proba(vectorized_text)[0] # Probabilities for each class
                
                # Get top 5 predictions
                # Argsort returns indices that would sort the array. [::-1] reverses it for descending order.
                top_indices = np.argsort(probabilities)[::-1][:5] 
                
                # Ensure encoder supports inverse_transform
                if not hasattr(self.encoder, 'inverse_transform'):
                    print("CONSOLE: Label encoder does not support inverse_transform.")
                    # If you can't inverse_transform, you might have to return raw indices or use fallback
                    # For now, assuming the fallback has textual roles
                    return self._fallback_role_prediction(resume_text)

                roles = self.encoder.inverse_transform(top_indices) # Convert integer labels back to role names
                scores = [round(probabilities[i] * 100, 2) for i in top_indices]
                
                return {"roles": roles.tolist(), "scores": scores}
            except Exception as e:
                print(f"CONSOLE: Prediction error with ML models: {e}")
                print("CONSOLE: Using fallback role prediction.")
        
        # Fallback if ML models not available or error during prediction
        return self._fallback_role_prediction(resume_text)

    def _fallback_role_prediction(self, resume_text: str) -> Dict:
        print("CONSOLE: Executing fallback role prediction.")
        # Using self.job_roles for a broader list of potential roles
        # More sophisticated keyword association might be needed.
        # This is a simplified example.
        role_keywords = {
            "Data Analyst": ["data", "analyst", "sql", "excel", "tableau", "power bi", "analysis", "reporting"],
            "Software Engineer": ["software", "engineer", "developer", "programming", "code", "agile", "java", "python", "c++"],
            "Frontend Developer": ["frontend", "ui", "ux", "html", "css", "javascript", "react", "angular", "vue"],
            "Backend Developer": ["backend", "api", "server", "database", "node.js", "python", "java", "microservices"],
            "Full Stack Developer": ["full stack", "fullstack", "frontend", "backend", "react", "node.js", "database"],
            "Machine Learning Engineer": ["machine learning", "ml", "ai", "tensorflow", "pytorch", "scikit-learn", "deep learning"],
            "DevOps Engineer": ["devops", "ci/cd", "docker", "kubernetes", "aws", "azure", "automation", "jenkins"],
            "Business Analyst": ["business analyst", "requirements", "stakeholder", "process improvement", "erp", "crm"],
            "Product Manager": ["product manager", "product owner", "roadmap", "user stories", "agile", "market research"],
            "UI/UX Designer": ["ui/ux", "designer", "figma", "sketch", "adobe xd", "user interface", "user experience", "wireframe"],
            "Data Scientist": ["data scientist", "statistics", "python", "r", "machine learning", "algorithms", "modeling"],
            "Cybersecurity Analyst": ["cybersecurity", "security analyst", "infosec", "siem", "firewall", "penetration testing"],
            "Mobile Developer": ["mobile developer", "ios", "android", "swift", "kotlin", "react native", "flutter"],
            "Cloud Engineer": ["cloud engineer", "aws", "azure", "gcp", "cloud architecture", "terraform", "kubernetes"],
            "System Administrator": ["sysadmin", "system administrator", "linux", "windows server", "networking", "vmware"]
        } # Use all roles from self.job_roles or refine keyword sets

        text_lower = self.clean_resume(resume_text).lower()
        role_scores = {role: 0 for role in self.job_roles} # Initialize all roles defined in system
        
        for role_name, keywords in role_keywords.items():
            if role_name in role_scores: # ensure the keyword set matches a role in self.job_roles
                for keyword in keywords:
                    if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                        role_scores[role_name] += 1
        
        # Sort by score
        sorted_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Filter out roles with zero scores, unless all have zero scores
        filtered_sorted_roles = [item for item in sorted_roles if item[1] > 0]
        if not filtered_sorted_roles and sorted_roles: # if all scores are 0, return top ones anyway
            filtered_sorted_roles = sorted_roles

        roles = [role for role, _ in filtered_sorted_roles[:5]] # Top 5
        # Simple scaling: Max score determines 100%, or just scale by number of keywords.
        # Let's say max possible keywords matched is around 5 for a strong match = 100%
        # For simplicity, cap at 100 if many keywords
        scores = [min(score * 20, 100) for _, score in filtered_sorted_roles[:5]]
        
        # Ensure we return something even if no keywords match for any role
        if not roles:
            return {"roles": ["General Application"], "scores": [20]} # Default if no matches
            
        return {"roles": roles, "scores": scores}

    def generate_job_suggestions(self, resume_data: Dict, preferences: Dict = None) -> List[Dict]:
        suggestions = []
        # Create a single text string from resume_data for role prediction
        resume_text_for_prediction = ' '.join(str(v) for k, v in resume_data.items() if k in ['skills', 'experience', 'education'])
        if not resume_text_for_prediction.strip() and 'full_text' in resume_data : # use full_text if available
            resume_text_for_prediction = resume_data['full_text']


        role_prediction = self.predict_job_roles(resume_text_for_prediction)
        
        resume_skills = resume_data.get('skills', [])
        if isinstance(resume_skills, str) : # if skills is a string, attempt to listify
             resume_skills = [skill.strip() for skill in resume_skills.split(',') if skill.strip()]
        if not resume_skills: resume_skills = ["various technologies"]


        for i, (role, score) in enumerate(zip(role_prediction["roles"][:5], 
                                            role_prediction["scores"][:5])):
            if score < 10 and i > 0: # Heuristic: if score is too low (and not the top one), maybe don't suggest
                 continue

            suggestion = {
                "role": role,
                "match_score": score, # This score comes from role_prediction
                "required_skills": self._get_role_requirements(role),
                "salary_range": self._estimate_salary_range(role, resume_data.get('experience', '0 years')),
                "recommendation_reason": f"Potential fit based on your profile. Highlight skills like {', '.join(resume_skills[:3])} for roles like this."
            }
            
            # Add more nuanced reason if possible
            matched_req_skills = [s for s in self._get_role_requirements(role) if s.lower() in (skill.lower() for skill in resume_skills)]
            if matched_req_skills:
                 suggestion["recommendation_reason"] = f"Matches your skills in {', '.join(matched_req_skills)}. Consider emphasizing these for the {role} role."
            elif resume_skills != ["various technologies"]:
                 suggestion["recommendation_reason"] = f"Your skills in {', '.join(resume_skills[:3])} could be relevant. Tailor your application to highlight alignment with {role} requirements."


            suggestions.append(suggestion)
        
        return suggestions

    def _get_role_requirements(self, role: str) -> List[str]:
        # These are illustrative. A real system would have a more extensive database.
        role_requirements = {
            "Data Analyst": ["SQL", "Excel", "Python for Data Analysis", "Tableau/Power BI", "Statistical Analysis"],
            "Software Engineer": ["Proficiency in a language (e.g., Java, Python, C++)", "Data Structures", "Algorithms", "Version Control (Git)", "Problem Solving"],
            "Frontend Developer": ["HTML", "CSS", "JavaScript", "React/Angular/Vue", "Responsive Design", "REST APIs"],
            "Backend Developer": ["Server-side language (e.g., Python/Java/Node.js)", "Databases (SQL/NoSQL)", "API Design (REST/GraphQL)", "Microservices Architecture"],
            "Full Stack Developer": ["Frontend skills (HTML, CSS, JS, Framework)", "Backend skills (Language, Database, API)", "DevOps basics"],
            "Machine Learning Engineer": ["Python", "Machine Learning Algorithms", "Deep Learning Frameworks (TensorFlow/PyTorch)", "Data Preprocessing", "Model Deployment"],
            "DevOps Engineer": ["CI/CD tools (Jenkins, GitLab CI)", "Containerization (Docker)", "Orchestration (Kubernetes)", "Cloud Platforms (AWS/Azure/GCP)", "Scripting (Bash/Python)"],
            "Business Analyst": ["Requirements Elicitation", "Data Analysis", "Process Modeling", "Stakeholder Management", "Agile/Scrum"],
            "Product Manager": ["Product Strategy", "User Research", "Roadmap Planning", "Agile Methodologies", "Communication Skills"],
            "UI/UX Designer": ["User Interface Design", "User Experience Principles", "Wireframing/Prototyping Tools (Figma, Sketch)", "User Research", "Visual Design"],
            "Data Scientist": ["Advanced Python/R", "Statistical Modeling", "Machine Learning", "Big Data Technologies (Spark)", "Data Visualization"],
            "Cybersecurity Analyst": ["Network Security", "SIEM Tools", "Vulnerability Assessment", "Incident Response", "Security Frameworks (NIST, ISO 27001)"],
            "Mobile Developer": ["iOS (Swift/Objective-C) or Android (Kotlin/Java)", "Mobile UI/UX Principles", "API Integration", "Version Control"],
            "Cloud Engineer": ["Cloud Platform Expertise (AWS/Azure/GCP)", "Infrastructure as Code (Terraform)", "Networking", "Security Best Practices", "Containerization/Orchestration"],
            "System Administrator": ["OS Management (Linux/Windows Server)", "Networking Fundamentals", "Scripting", "Virtualization", "Hardware/Software Troubleshooting"]
        }
        return role_requirements.get(role, ["Strong Problem Solving", "Good Communication", "Teamwork Ability"])


    def _estimate_salary_range(self, role: str, experience: str) -> str:
        base_salaries = {
            "Data Analyst": 60000, "Software Engineer": 80000, "Frontend Developer": 75000,
            "Backend Developer": 85000, "Full Stack Developer": 90000, 
            "Machine Learning Engineer": 95000, "DevOps Engineer": 90000, 
            "Business Analyst": 70000, "Product Manager": 100000, "UI/UX Designer": 70000,
            "Data Scientist": 100000, "Cybersecurity Analyst": 85000,
            "Mobile Developer": 80000, "Cloud Engineer": 95000, "System Administrator": 65000
        } # More comprehensive list
        default_base = 70000
        base = base_salaries.get(role, default_base)
        
        exp_years = 0
        if isinstance(experience, str):
            # More robust extraction of years from "X years" or "X+ years"
            exp_match = re.search(r'(\d+\.?\d*|\d+)', experience) # Catches "5", "5.5", "10+" -> "10"
            if exp_match:
                try:
                    exp_years = float(exp_match.group(1))
                except ValueError:
                    exp_years = 0 # Default if conversion fails
        elif isinstance(experience, (int, float)):
            exp_years = experience

        # Experience multiplier - non-linear could be better but simple for now
        if exp_years <= 1:
            multiplier = 1.0
        elif exp_years <= 3:
            multiplier = 1.1 + (exp_years -1) * 0.1 # e.g. 2yrs=1.2, 3yrs=1.3
        elif exp_years <= 5:
            multiplier = 1.3 + (exp_years -3) * 0.075 # e.g. 4yrs=1.375, 5yrs=1.45
        elif exp_years <= 10:
            multiplier = 1.45 + (exp_years-5) * 0.05 # e.g. 10yrs = 1.7
        else: # 10+ years
            multiplier = 1.7 + (exp_years-10) * 0.03 # Slower growth after 10 years

        adjusted_salary_low = int(base * multiplier)
        
        # Define a salary range width, maybe % of base or fixed
        range_width_percentage = 0.15 # e.g. 15% of adjusted_salary_low for the range
        range_width_absolute = max(15000, int(adjusted_salary_low * range_width_percentage)) # Min 15k width

        adjusted_salary_high = adjusted_salary_low + range_width_absolute

        # Round to nearest $1000 or $5000 for cleaner look
        adjusted_salary_low = int(round(adjusted_salary_low / 1000) * 1000)
        adjusted_salary_high = int(round(adjusted_salary_high / 1000) * 1000)

        return f"${adjusted_salary_low:,} - ${adjusted_salary_high:,} per year (estimated)"

    def analyze_resume_complete(self, resume_text: str, job_description: str = None, 
                              preferences: Dict = None) -> Dict:
        if not resume_text or not resume_text.strip():
             return {
                "timestamp": datetime.now().isoformat(),
                "error": "Resume text is empty. Please provide resume content.",
                "parsed_resume": {}, "similarity_scores": {}, 
                "role_predictions": {"roles": [], "scores": []}, 
                "job_suggestions": [], "analysis_summary": "Analysis failed due to empty resume."
            }
        
        # Add the full resume text to parsed_data so it can be used later if needed, e.g., by generate_job_suggestions
        parsed_data = self.parse_resume(resume_text)
        parsed_data['full_text'] = resume_text # Store the original resume text

        similarity_scores = {}
        if job_description and job_description.strip():
            similarity_scores = self.calculate_similarity_scores(resume_text, job_description)
        
        role_predictions = self.predict_job_roles(resume_text) # Use full resume text for prediction
        job_suggestions = self.generate_job_suggestions(parsed_data, preferences) # Pass all parsed_data
        
        name_for_summary = parsed_data.get('name', 'Candidate')
        if name_for_summary == "Not found": name_for_summary = "Candidate"

        complete_analysis = {
            "timestamp": datetime.now().isoformat(),
            "parsed_resume": parsed_data,
            "similarity_scores": similarity_scores,
            "role_predictions": role_predictions,
            "job_suggestions": job_suggestions,
            "analysis_summary": f"Analysis completed for {name_for_summary}."
        }
        
        return complete_analysis
    
# --- end of dummy.py ---

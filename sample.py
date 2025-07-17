# --- contents of sample.py ---
from dummy import JobMatchingSystem # Import the class from dummy.py
import json                         # For pretty printing the dictionary
import os
import pickle
# Import necessary sklearn components for creating dummy .pkl files if needed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def create_dummy_pkl_files_if_needed():
    """Creates dummy .pkl files for the system to run if they don't exist."""
    # Dummy data for creating placeholder .pkl files
    dummy_texts = ["this is a sample developer resume", "another sample analyst resume", "yet another engineer document"]
    # Make sure dummy_labels are a subset of the JobMatchingSystem().job_roles or handled by encoder.fit()
    jms_instance_for_roles = JobMatchingSystem() # Temporary instance to get job_roles
    all_possible_roles_for_encoder = list(set(jms_instance_for_roles.job_roles + ["Software Engineer", "Data Analyst"]))


    # Assign some dummy labels, ensure they are in the all_possible_roles_for_encoder
    dummy_labels = []
    for i, text in enumerate(dummy_texts):
        if "developer" in text or "engineer" in text:
            dummy_labels.append("Software Engineer")
        elif "analyst" in text:
            dummy_labels.append("Data Analyst")
        else: # Default fallback
            dummy_labels.append(all_possible_roles_for_encoder[0])


    if not os.path.exists('tfidf.pkl'):
        print("Creating dummy tfidf.pkl...")
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        vectorizer.fit(dummy_texts + ["random text to ensure model is not empty"]) # Add some text to avoid empty vocab
        with open('tfidf.pkl', 'wb') as f: pickle.dump(vectorizer, f)
    
    if not os.path.exists('encoder.pkl'):
        print("Creating dummy encoder.pkl...")
        encoder = LabelEncoder()
        encoder.fit(all_possible_roles_for_encoder)
        with open('encoder.pkl', 'wb') as f: pickle.dump(encoder, f)

    if not os.path.exists('clf.pkl'):
        print("Creating dummy clf.pkl...")
        # We need to use the fitted vectorizer and encoder on the dummy data
        try:
            with open('tfidf.pkl', 'rb') as f: temp_vectorizer = pickle.load(f)
            with open('encoder.pkl', 'rb') as f: temp_encoder = pickle.load(f)
        except FileNotFoundError:
            print("ERROR: Could not load dummy tfidf.pkl or encoder.pkl to create clf.pkl. Please ensure they are created.")
            return # Cannot proceed

        # Check if vectorizer is fitted
        if not hasattr(temp_vectorizer, 'vocabulary_') or not temp_vectorizer.vocabulary_:
             print("Dummy TFIDF vectorizer is not fitted. Fitting it now.")
             temp_vectorizer.fit(dummy_texts + ["ensure not empty vocab"])


        X_transformed = temp_vectorizer.transform(dummy_texts)
        
        # Ensure labels are known to the encoder
        valid_dummy_labels = [label for label in dummy_labels if label in temp_encoder.classes_]
        if len(valid_dummy_labels) < len(dummy_labels):
            print(f"Warning: Some dummy labels were not in the encoder's classes. Using only valid ones.")
        if not valid_dummy_labels : # If no valid labels, create at least one default
            print("No valid dummy labels for classifier training. Using a default label.")
            valid_dummy_labels = [temp_encoder.classes_[0]] # Use first known class
            # Adjust X_transformed to match, e.g., use first text sample
            X_transformed = temp_vectorizer.transform([dummy_texts[0]])


        y_transformed = temp_encoder.transform(valid_dummy_labels)
        
        classifier = RandomForestClassifier(random_state=42, n_estimators=10) # n_estimators=10 for faster dummy training

        if X_transformed.shape[0] >= 1 and len(np.unique(y_transformed)) >=1 :
             try:
                # Ensure there are at least 2 samples if y has more than one unique class (for some classifiers)
                # For RandomForest with 1 sample and 1 class, it's usually fine.
                if X_transformed.shape[0] < 2 and len(np.unique(y_transformed)) > 1:
                    # Duplicate data to meet minimum sample requirements if needed by stricter model types
                    # This is a HACK for dummy purposes. Real models need real data.
                    X_transformed_fit = np.vstack([X_transformed, X_transformed])
                    y_transformed_fit = np.concatenate([y_transformed, y_transformed])
                    print("Warning: Duplicating dummy data to attempt classifier fit.")
                else:
                    X_transformed_fit = X_transformed
                    y_transformed_fit = y_transformed

                classifier.fit(X_transformed_fit, y_transformed_fit)
             except ValueError as e:
                print(f"Warning: Could not fit dummy classifier due to data constraints: {e}")
                # Create a mock classifier if fitting fails
                class MockClassifier:
                    def predict(self, X): return np.array([temp_encoder.transform([all_possible_roles_for_encoder[0]])[0]] * X.shape[0])
                    def predict_proba(self, X): 
                        probs = np.zeros((X.shape[0], len(temp_encoder.classes_)))
                        if probs.shape[1] > 0: probs[:,np.where(temp_encoder.classes_ == all_possible_roles_for_encoder[0])[0][0]] = 1.0 
                        return probs
                classifier = MockClassifier()
                print("Created a MockClassifier as a fallback.")
        else:
            class MockClassifier:
                def predict(self, X): return np.array([temp_encoder.transform([all_possible_roles_for_encoder[0]])[0]] * X.shape[0])
                def predict_proba(self, X): 
                    probs = np.zeros((X.shape[0], len(temp_encoder.classes_)))
                    if probs.shape[1] > 0: probs[:,np.where(temp_encoder.classes_ == all_possible_roles_for_encoder[0])[0][0]] = 1.0 
                    return probs
            classifier = MockClassifier()
            print("Warning: Dummy classifier created with mock due to insufficient distinct samples/labels.")
            
        with open('clf.pkl', 'wb') as f: pickle.dump(classifier, f)
    print("Dummy .pkl file check complete.")


def main_cli():
    # --- IMPORTANT ---
    # Create dummy .pkl files for this CLI example to run if real ones are missing.
    # If you have your actual .pkl files, this function will just skip creation.
    create_dummy_pkl_files_if_needed()
    # --- End of dummy .pkl creation check ---

    # Now initialize the system. It will try to load the .pkl files.
    jms = JobMatchingSystem()

    sample_resume_text = """
    John Doe
    john.doe@email.com | (123) 456-7890

    Summary
    Experienced Python Developer with 5 years of experience in web development and data analysis. 
    Proficient in Django, Flask, Pandas, and SQL. Seeking a software engineer role.

    Skills
    Programming: Python, Java, JavaScript
    Web Development: Django, Flask, HTML, CSS
    Data: Pandas, NumPy, SQL, Matplotlib
    Tools: Git, Docker

    Experience
    Senior Software Developer | Tech Solutions Inc. | 2019 - Present
    - Developed web applications using Python and Django.
    - Performed data analysis tasks using Pandas.

    Education
    Master of Science in Computer Science | XYZ University | 2019
    Bachelor of Science in Information Technology | ABC College | 2017
    """

    sample_job_description = """
    We are looking for a Python Developer (Software Engineer) to join our dynamic team.
    Responsibilities include designing and developing backend services and working closely with databases.
    Required skills: Python, Django or Flask framework, SQL database. 3+ years of commercial experience.
    """

    print("\n--- Analyzing Resume ---")
    # Set job_description to None if you don't want to compare with a specific job
    analysis_results = jms.analyze_resume_complete(
        resume_text=sample_resume_text,
        job_description=sample_job_description  # Or set to None
    )

    print("\n--- Analysis Results (CLI) ---")
    # Pretty print the JSON results
    print(json.dumps(analysis_results, indent=4))

    # Example of accessing specific parts:
    print(f"\nPredicted Top Role: {analysis_results.get('role_predictions', {}).get('roles', ['N/A'])[0]}")
    if sample_job_description:
        print(f"Overall Match Score with JD: {analysis_results.get('similarity_scores', {}).get('combined_score', 'N/A')}%")

if __name__ == "__main__":
    main_cli()

# --- end of sample.py ---
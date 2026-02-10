import streamlit as st
import spacy
import re
from spacy.cli import download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    nlp = spacy.load("en_core_web_sm")
except:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

resumes = {
    "Amit Sharma": "Data Analyst skilled in Python SQL Excel Power BI",
    "Neha Verma": "Software Engineer with Java and Spring Boot",
    "Rahul Singh": "Business Analyst skilled in Excel SQL Tableau"
}

job_description = """
Looking for a Data Analyst with Python, SQL, Excel, Power BI skills
"""

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', ' ', text)
    doc = nlp(text)
    return " ".join([t.lemma_ for t in doc if not t.is_stop])

def calculate_score(resume, jd):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume, jd])
    return round(cosine_similarity(vectors[0], vectors[1])[0][0] * 100, 2)

st.title("🤖 AI Resume Screening Bot")

results = []
for name, resume in resumes.items():
    score = calculate_score(clean_text(resume), clean_text(job_description))
    results.append((name, score))

results.sort(key=lambda x: x[1], reverse=True)

for r in results:
    st.write(f"**{r[0]}** → {r[1]}% match")


import streamlit as st
import difflib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK stopwords
nltk.download('stopwords')

# Load and preprocess dataset
df = pd.read_csv('Online_Courses.csv')

# Rename columns
df.rename(columns={
    "Description": "Short Intro",
    "Skills / Interests": "Skills"
}, inplace=True)

# Fill missing values
df.fillna('', inplace=True)

# Generate textual representation for each course
def textual_rep(row):
    return f"""
    Title: {row['Title']}
    URL: {row['URL']}
    Category: {row['Category']}
    Sub-Category: {row['Sub-Category']}
    Language: {row['Language']}
    Skills: {row['Skills']}
    Description: {row['Short Intro']}
    """

df['textual_representation'] = df.apply(textual_rep, axis=1)

# Text preprocessing
porter_stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content).lower()
    words = content.split()
    return ' '.join(porter_stemmer.stem(word) for word in words if word not in stop_words)

df['processed_text'] = df['textual_representation'].apply(stemming)

# Vectorize the processed text
vectorizer = TfidfVectorizer()
text_vectors = vectorizer.fit_transform(df['processed_text'])

# Precompute similarity matrix
similarity_matrix = cosine_similarity(text_vectors)

# Streamlit app
st.title('Course Recommendation')
user_input = st.text_input('Enter What You Want to Learn:')

if user_input:
    # Preprocess user input and compute similarity
    user_input_processed = stemming(user_input)
    user_vector = vectorizer.transform([user_input_processed])
    similarity_scores = cosine_similarity(user_vector, text_vectors).flatten()

    # Get top recommendations
    top_indices = similarity_scores.argsort()[-10:][::-1]
    top_scores = similarity_scores[top_indices]

    # Display recommendations
    if top_scores[0] > 0:
        st.subheader('Courses Suggested for You:')
        for idx, score in zip(top_indices, top_scores):
            title = df.loc[idx, 'Title']
            url = df.loc[idx, 'URL']
            category = df.loc[idx, 'Sub-Category']
            skills = df.loc[idx, 'Skills']
            
            # Display course details
            st.markdown(f"### [{title}]({url})")
            st.write(f"**Category:** {category}")
            st.write(f"**Skills:** {skills}")
            st.write(f"**Relevance Score:** {score:.2f}")
            st.write("---")
    else:
        st.write("No relevant courses found. Please refine your search.")

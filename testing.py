import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Load large data from JSON file
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Preprocess texts using spaCy
def preprocess_texts(texts, nlp):
    processed_texts = []
    for text in texts:
        doc = nlp(text.lower())
        processed_texts.append(" ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct]))
    return processed_texts

# Vectorize user query
def vectorize_query(query, vectorizer):
    return vectorizer.transform([query])

# Find most relevant results
def find_relevant_results(query_vector, data_vectors, data, top_n=3):
    similarities = cosine_similarity(query_vector, data_vectors)
    top_indices = similarities.argsort()[0][-top_n:][::-1]
    return [data[i] for i in top_indices]

# Main function
def main():
    # Load data
    data = load_data("myfile.json")
    
    # Extract texts and chapters from data and build book lookup
    texts = [hadith['text'] for hadith in data['hadiths']]
    book_lookup = {hadith['hadithnumber']: data['metadata']['name'] for hadith in data['hadiths']}
    chapter_lookup = {int(chapter_num): chapter_name for chapter_num, chapter_name in data['metadata']['sections'].items()}
    
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")
    
    # Preprocess the texts
    processed_texts = preprocess_texts(texts, nlp)
    
    # Vectorize the texts
    vectorizer = TfidfVectorizer()
    data_vectors = vectorizer.fit_transform(processed_texts)
    
    # Take user query
    user_query = input("Enter your query: ").strip()
    
    if user_query:
        # Preprocess and vectorize user query
        processed_query = preprocess_texts([user_query], nlp)[0]
        query_vector = vectorize_query(processed_query, vectorizer)
        # Find relevant results
        relevant_results = find_relevant_results(query_vector, data_vectors, data['hadiths'])
        # Print relevant results
        for result in relevant_results:
            print("\nHadith Number:", result['hadithnumber'])
            print("Chapter:", chapter_lookup.get(result['reference']['book'], 'Unknown Chapter'))
            print("Hadith:", result['text'])
            print("Book:", book_lookup[result['hadithnumber']])
    else:
        print('Please Enter your query!')

if __name__ == "__main__":
    main()

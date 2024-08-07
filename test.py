import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Load data from JSON file
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Preprocess texts using spaCy for better tokenization and lemmatization
def preprocess_texts(texts, nlp):
    processed_texts = []
    for text in texts:
        doc = nlp(text.lower())
        processed_texts.append(" ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct]))
    return processed_texts

# Vectorize user query
def vectorize_query(query, vectorizer):
    return vectorizer.transform([query])

# Find most relevant results based on cosine similarity
def find_relevant_results(query_vector, data_vectors, data, top_n=3):
    similarities = cosine_similarity(query_vector, data_vectors)
    top_indices = similarities.argsort()[0][-top_n:][::-1]
    return [data[i] for i in top_indices]

# Main function to run the search
def main():
    # Load data
    data = load_data("myfile.json")
    
    # Extract texts and metadata from data
    texts = [hadith['text'] for hadith in data['hadiths']]
    book_lookup = {hadith['hadithnumber']: data['metadata']['name'] for hadith in data['hadiths']}
    chapter_lookup = {int(chapter_num): chapter_name for chapter_num, chapter_name in data['metadata']['sections'].items()}
    
    # Initialize spaCy for text preprocessing accuracy maintain rkhne ke liye
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    
    # Preprocess the texts using spaCy
    processed_texts = preprocess_texts(texts, nlp)
    
    # Vectorize the preprocessed texts
    vectorizer = TfidfVectorizer()
    data_vectors = vectorizer.fit_transform(processed_texts)
    
    # Take user query
    user_query = input("Enter your query: ").strip()
    
    if user_query:
        # Preprocess and vectorize user query
        processed_query = " ".join([token.lemma_ for token in nlp(user_query.lower()) if not token.is_stop and not token.is_punct])
        query_vector = vectorize_query(processed_query, vectorizer)
        
        # Find relevant results based on cosine similarity
        relevant_results = find_relevant_results(query_vector, data_vectors, data['hadiths'])
        
        # Print relevant results
        print("\nTop relevant Hadiths for your query:\n")
        for result in relevant_results:
            print(f"Hadith Number: {result['hadithnumber']}")
            print(f"Chapter: {chapter_lookup.get(result['reference']['book'], 'Unknown Chapter')}")
            print(f"Hadith: {result['text']}")
            print(f"Book: {book_lookup[result['hadithnumber']]}\n")
    else:
        print('Please enter your query!')

if __name__ == "__main__":
    main()

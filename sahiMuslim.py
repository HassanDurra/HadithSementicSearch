from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

app = Flask(__name__)
CORS(app)
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

# Initialize spaCy and load data
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
data = load_data("sahi_muslim.json")

# Extract texts and metadata from data
texts = [hadith['text'] for hadith in data['hadiths']]
book_lookup = {hadith['hadithnumber']: data['metadata']['name'] for hadith in data['hadiths']}
chapter_lookup = {int(chapter_num): chapter_name for chapter_num, chapter_name in data['metadata']['sections'].items()}

# Preprocess the texts using spaCy
processed_texts = preprocess_texts(texts, nlp)

# Vectorize the preprocessed texts
vectorizer = TfidfVectorizer()
data_vectors = vectorizer.fit_transform(processed_texts)

@app.route('/search/sahi_muslim', methods=['GET'])
def search():
    user_query = request.args.get('query', '').strip()
    user_book  = request.args.get('book_name' ,'').strip();
    if not user_query:
        return jsonify({'error': 'Please provide a query parameter!'}), 400

    # Preprocess and vectorize user query
    processed_query = " ".join([token.lemma_ for token in nlp(user_query.lower()) if not token.is_stop and not token.is_punct])
    query_vector = vectorize_query(processed_query, vectorizer)

    # Find relevant results based on cosine similarity
    relevant_results = find_relevant_results(query_vector, data_vectors, data['hadiths'])

    # Format the results for the response
    response = []
    for result in relevant_results:
        response.append({
            'message':'success',
            'hadith_number': result['hadithnumber'],
            'chapter': chapter_lookup.get(result['reference']['book'], 'Unknown Chapter'),
            'hadith': result['text'],
            'book': book_lookup[result['hadithnumber']]
        })

    return jsonify({'data':response , 'message':'success'})

if __name__ == "__main__":
    app.run(debug=True)

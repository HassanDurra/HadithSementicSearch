from flask import Flask, request, jsonify
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
# Load large data from JSON file
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Vectorize user query
def vectorize_query(query, vectorizer):
    return vectorizer.transform([query])

# Find most relevant results
def find_relevant_results(query_vector, data_vectors, data, top_n=3):
    similarities = cosine_similarity(query_vector, data_vectors)
    top_indices = similarities.argsort()[0][-top_n:][::-1]
    return [data[i] for i in top_indices]

# Load data and set up vectorizer and data_vectors globally
data = load_data("myfile.json")
texts = [hadith['text'] for hadith in data['hadiths']]
book_lookup = {hadith['hadithnumber']: data['metadata']['name'] for hadith in data['hadiths']}
chapter_lookup = {int(chapter_num): chapter_name for chapter_num, chapter_name in data['metadata']['sections'].items()}
texts = [hadith['text'] for hadith in data['hadiths']]
vectorizer = TfidfVectorizer()
data_vectors = vectorizer.fit_transform(texts)

@app.route('/api/search/<query>', methods=['GET'])
def search(query):
    print(query)
    query_vector = vectorize_query(query, vectorizer)
    relevant_results = find_relevant_results(query_vector, data_vectors, data['hadiths'])
    for result in relevant_results:
        print("\nHadith Number:", result['hadithnumber'])
        print("Chapter:", chapter_lookup.get(result['reference']['book'], 'Unknown Chapter'))
        print("Hadith:", result['text'])
        print("Book:", book_lookup[result['hadithnumber']])
        response = []
    for result in relevant_results:
        response.append({
            'hadith_number': result['hadithnumber'],
            'chapter': chapter_lookup.get(result['reference']['book'], 'Unknown Chapter'),
            'hadith': result['text'],
            'book': book_lookup[result['hadithnumber']]
        })

    return jsonify({'results': response})


if __name__ == "__main__":
    app.run(debug=True)

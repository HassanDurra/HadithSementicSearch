import json
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1 : creating a function to retreive data from json
def load_data(file_path):
    with open(file_path , 'r' , encoding='utf-8') as file:
        data = json.load(file)
        return data
 
# Step 2 : Creating a function to vectorize the Query
def vectorize_query(query , vectorizer):
    return vectorizer.transform([query])

# Step 3 : Creating a function to retrieve relevent Sementic Results

def find_relevent_results(query_vector , data_vectors , data , top_n = 3):  #I used top_n to retreive the total of 3 most relevant Results
    similarities  = cosine_similarity(query_vector , data_vectors)
    top_indices = similarities.argsort()[0][-top_n:][::-1]
    return [data[i] for i in top_indices]

# Creating the main function
def main():
    
    data = load_data('myfile.json') # Add your json file path 
    #Extract texts from data
    texts = [item['text'] for item in data]
    
    #Vectorize the texts
    vectorizer = TfidfVectorizer()
    data_vectors = vectorizer.fit_transform(texts)

    # Take User query from here
    user_query = input('Enter your Query:').strip()
    
    if user_query:
        # Vectorizing User query from here
        query_vector  = vectorize_query(user_query , vectorizer)
        # Finding Most relevant Results:
        relevant_results = find_relevent_results (query_vector , data_vectors , data)
        # printing the results
        for result in relevant_results:
            print("\nHadith Number:" , result['hadithnumber'])
            print("Hadith:" , result['text'])
    else:
        print("Pease Enter your Query!") 
if __name__ == "__main__":
    main()          
    
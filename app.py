from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import joblib
import os

app = Flask(__name__, static_folder='static', template_folder='templates')


# Load model and data
try:
    model = joblib.load('model.h5')
    book_pivot = joblib.load('book_pivot.pkl')
    print("Model and data loaded successfully!")
except Exception as e:
    print(f"Error loading model or data: {str(e)}")
    print("Please run train_model.py first to generate model files")
    exit(1)

@app.route('/')
def home():
    book_titles = list(book_pivot.index)
    return render_template('index.html', book_titles=book_titles)

# Add static file serving
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    book_title = request.form.get('book_title', '') if request.method == 'POST' else request.args.get('title', '')
    
    recommended_books = get_recommends(book_title)
    
    if request.method == 'POST':
        return render_template('index.html', 
                            recommendation=recommended_books,
                            book_titles=list(book_pivot.index))
    return jsonify(recommended_books)

def get_recommends(book=""):
    try:
        book_index = np.where(book_pivot.index == book)[0][0]
        distances, indices = model.kneighbors(
            book_pivot.iloc[book_index, :].values.reshape(1, -1), 
            n_neighbors=6)
        
        similar_books = [
            [book_pivot.index[indices.flatten()[i]], float(distances.flatten()[i])]
            for i in range(1, len(indices.flatten()))
        ]
        
        similar_books_sorted = sorted(similar_books, key=lambda x: x[1], reverse=False)
        return [book, similar_books_sorted]
    except Exception as e:
        print(f"Error in recommendation: {str(e)}")
        return [book, []]

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

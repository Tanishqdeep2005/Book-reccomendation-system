# Book Recommendation System

A content-based book recommendation system built using Python and Streamlit.  
The application suggests similar books based on textual similarity computed using TF-IDF and cosine similarity. Users can search for books through an interactive dropdown and access direct links to purchase books and read reviews using ISBN-based redirection.

## Features
- Searchable dropdown to select books by title  
- Content-based recommendations using TF-IDF and cosine similarity  
- Displays book title, author, and average rating  
- Direct links to purchase books via Amazon using ISBN  
- Direct links to view reviews via Goodreads  
- Web-based dashboard built with Streamlit  

## Tech Stack
- Python  
- Pandas  
- Scikit-learn  
- NumPy  
- Streamlit  

## Project Structure
Book-Recommender-System/  
│  
├── app.py  
├── books_cleaned.csv  
├── requirements.txt  
├── README.md  
└── .gitignore  

## Installation

1. Clone the repository:
   git clone https://github.com/your-username/book-recommender-system.git  
   cd book-recommender-system  

2. Install dependencies:
   pip install -r requirements.txt  

3. Run the application:
   streamlit run app.py  

## How It Works
1. The dataset is loaded and relevant text fields (title, authors, and content) are combined.  
2. TF-IDF vectorization converts the text into numerical feature vectors.  
3. Cosine similarity is computed between all books to measure content similarity.  
4. When a user selects a book, the system recommends the most similar books based on similarity scores.  
5. ISBN values are used to generate external links for purchasing and reviewing books.  

## Dataset
The dataset contains the following key fields:
- title  
- authors  
- average_rating  
- isbn  
- language_code  
- content  


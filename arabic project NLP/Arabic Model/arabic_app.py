import streamlit as st
import pickle
import time
from joblib import load
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from bidi.algorithm import get_display
from arabic_reshaper import reshape
import pandas as pd

# Load the models
model = load(r'C:\Users\Asus\Desktop\arabic project NLP\Arabic Model\svm_text_classification_model.pkl')
vectorizer = load(r'C:\Users\Asus\Desktop\arabic project NLP\Arabic Model\tfidf_vectorizer.pkl')

# Initialize session state to track predictions
if 'sentiment_counts' not in st.session_state:
    st.session_state.sentiment_counts = {'Positive': 0, 'Negative': 0}

# Streamlit app title
st.title('Arabic Review Sentiment Analysis')

# Input field for the review
tweet = st.text_input('Enter your Review')

# Button to trigger prediction
submit = st.button('Predict')

if submit:
    start = time.time()
    # Transform the review using the TF-IDF vectorizer
    tweet_transformed = vectorizer.transform([tweet])
    
    # Predict sentiment using the SVM model
    prediction = model.predict(tweet_transformed)
    end = time.time()
    
    # Update session state counts
    if prediction[0] == 2:
        sentiment_label = 'Positive'
        st.session_state.sentiment_counts['Positive'] += 1
        st.write('Positive Review 😁😁')
    else:
        sentiment_label = 'Negative'
        st.session_state.sentiment_counts['Negative'] += 1
        st.write('Negative Review 😡😡')

    st.write('Prediction time taken: ', round(end - start, 2), 'seconds')

    # Generate a word cloud from the review text
    reshaped_text = reshape(tweet)
    bidi_text = get_display(reshaped_text)
    wordcloud = WordCloud(
        font_path='arial.ttf',  # Replace with a valid Arabic font
        width=800,
        height=400,
        background_color='white'
    ).generate(bidi_text)

    st.subheader('Word Cloud of Your Review')
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# Sidebar: Display dynamic visualization of sentiment distribution
st.sidebar.header("Sentiment Distribution")
if st.sidebar.button('Show Sentiment Distribution'):
    st.subheader('Dynamic Sentiment Distribution')
    sentiment_counts = st.session_state.sentiment_counts

    # Convert to DataFrame for visualization
    sentiment_df = pd.DataFrame(list(sentiment_counts.items()), columns=['Sentiment', 'Count'])
    
    # Display bar chart
    st.bar_chart(sentiment_df.set_index('Sentiment'))

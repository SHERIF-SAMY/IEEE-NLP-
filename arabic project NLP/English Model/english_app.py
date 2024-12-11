# # install streamlit: pip install streamlit
# # run: streamlit run app.py

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load
import time
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

# Load the model and tokenizer
model = load_model(r'C:\Users\Asus\Desktop\arabic project NLP\English Model\English_review_model.h5')
tokenizer = load(r'C:\Users\Asus\Desktop\arabic project NLP\English Model\tokenizer.pkl')

# Initialize session state for sentiment tracking
if 'sentiment_counts' not in st.session_state:
    st.session_state.sentiment_counts = {'Positive': 0, 'Negative': 0}

# Streamlit app title
st.title('English Review Sentiment Analysis')

# Input field for the review
tweet = st.text_input('Enter your tweet')

# Button to trigger prediction
submit = st.button('Predict')

if submit:
    if not tweet.strip():
        st.error('Please enter a valid tweet.')
    else:
        start = time.time()
        
        # Preprocess input
        sequences = tokenizer.texts_to_sequences([tweet])
        padded_sequences = pad_sequences(sequences, maxlen=200)  # Correct maxlen
        
        # Predict sentiment
        prediction = model.predict(padded_sequences)
        end = time.time()
        
        # Display prediction time
        st.write('Prediction time taken: ', round(end - start, 2), 'seconds')
        
        # Determine sentiment
        if prediction[0] > 0.5:
            sentiment = 'Positive'
            st.session_state.sentiment_counts['Positive'] += 1
            st.write('Positive Review 😁😁')
        else:
            sentiment = 'Negative'
            st.session_state.sentiment_counts['Negative'] += 1
            st.write('Negative Review 😡😡')
        
        # Generate a word cloud from the tweet
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(tweet)
        st.subheader('Word Cloud of Your Tweet')
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

# Sidebar: Dynamic visualization of sentiment distribution
st.sidebar.header("Sentiment Distribution")
if st.sidebar.button('Show Sentiment Distribution'):
    st.subheader('Dynamic Sentiment Distribution')
    sentiment_counts = st.session_state.sentiment_counts

    # Convert to DataFrame for visualization
    sentiment_df = pd.DataFrame(list(sentiment_counts.items()), columns=['Sentiment', 'Count'])
    
    # Display bar chart
    st.bar_chart(sentiment_df.set_index('Sentiment'))
    
    # Pie chart
    st.subheader("Pie Chart of Sentiments")
    fig, ax = plt.subplots()
    ax.pie(
        sentiment_counts.values(), 
        labels=sentiment_counts.keys(), 
        autopct='%1.1f%%', 
        startangle=90, 
        colors=['green', 'red']
    )
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

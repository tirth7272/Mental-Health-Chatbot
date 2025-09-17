# app.py
import streamlit as st
import random
import json
import torch
from models.model import NeuralNet
from models.nltk_utils import tokenize, stem, bag_of_words

# Load intents from data/intents.json
with open('data\intents.json', 'r') as file:
    intents = json.load(file)

# Load the pre-trained model and other data from data/data.pth
data = torch.load('data\data.pth')

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# Function to process user input and generate responses
def get_response(message):
    # Tokenize the message
    sentence = tokenize(message)
    # Create a bag of words representation
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)
    # Make prediction using the model
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Check if prediction probability is high enough
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                return random.choice(intent['responses'])
    else:
        return "I'm sorry, I'm not sure how to respond to that."

# Streamlit UI
def main():
    st.title("Chat with Sam")

    # Text input for user to enter messages
    user_input = st.text_input("You:", "")

    if st.button("Send"):
        # Display user message
        st.text_area("You:", user_input, height=100)

        # Get and display bot response
        bot_response = get_response(user_input)
        st.text_area("Sam:", bot_response, height=100)

if __name__ == "__main__":
    main()

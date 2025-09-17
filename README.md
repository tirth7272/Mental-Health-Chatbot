🧠 Mental Health Chatbot

A Streamlit-based Mental Health Chatbot that uses PyTorch and NLTK to provide supportive conversations. The chatbot can recognize different intents like greetings, coping strategies, mental health concerns, and more, and reply with empathetic responses.

⚠️ Disclaimer: This chatbot is not a substitute for professional medical advice, diagnosis, or treatment. Always seek professional help if needed.

🚀 Features

💬 Predefined intents for greetings, mental health concerns, coping strategies, treatments, and more

🤖 Built using PyTorch with a simple feed-forward neural network

🔤 Natural Language Processing (NLP) with NLTK (tokenization, stemming, Bag of Words)

🌐 Interactive web interface built with Streamlit

🎯 Provides supportive and empathetic responses based on user input

🛠️ Tech Stack

Frontend: Streamlit

Backend: Python, PyTorch

NLP: NLTK

Dataset: Custom JSON intents (intents.json)

Other Libraries: NumPy, Random

📂 Project Structure
Mental-Health-Chatbot/
│── app.py              # Streamlit app for chatbot UI
│── train.py            # Training script for neural network
│── data/
│   ├── intents.json    # Dataset of intents, patterns, and responses
│   ├── data.pth        # Trained model weights
│── models/
│   ├── model.py        # Neural network model definition
│   ├── nltk_utils.py   # NLP utilities (tokenization, stemming, BoW)
│── requirements.txt    # Dependencies
│── README.md           # Project documentation

📂 Installation

Clone the repository:

git clone https://github.com/your-username/Mental-Health-Chatbot.git
cd Mental-Health-Chatbot


Install dependencies:

pip install -r requirements.txt --user


Train the model (optional):

python train.py


This will generate data/data.pth with trained model weights.

Run the chatbot:

streamlit run app.py



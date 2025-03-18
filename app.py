from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import pickle
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from textblob import TextBlob
from community import community_louvain

# Initialize Flask app
app = Flask(__name__)
CORS(app)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained model
MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model.pkl")
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# Load dataset and network edges
DATA_PATH = os.path.join(BASE_DIR, "data", "fake_accounts_dataset.csv")
EDGES_PATH = os.path.join(BASE_DIR, "data", "network_connections.csv")

df = pd.read_csv(DATA_PATH)
edges = pd.read_csv(EDGES_PATH)

# Initialize vectorizers and scalers
vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
count_vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=50)
scaler = StandardScaler()

vectorizer.fit(df['bio'].astype(str))
count_vectorizer.fit(df['bio'].astype(str))
scaler.fit(df[['followers', 'following', 'posts']])

# Create network graph
graph = nx.from_pandas_edgelist(edges, 'source', 'target', create_using=nx.Graph())
try:
    communities = community_louvain.best_partition(graph)
    betweenness = nx.betweenness_centrality(graph)
    eigenvector = nx.eigenvector_centrality(graph, max_iter=1000)
except Exception as e:
    print(f"⚠️ Graph feature computation error: {e}")
    communities, betweenness, eigenvector = {}, {}, {}

@app.route('/')
def home():
    return render_template('index.html')

def preprocess_input(data):
    required_fields = ['bio', 'followers', 'following', 'posts']
    if not all(field in data for field in required_fields):
        return None, {"error": "Invalid input. Missing required fields."}
    
    bio_text = data['bio']
    followers = int(data['followers'])
    following = int(data['following'])
    posts = int(data['posts'])
    account_id = data.get('account_id', -1)

    print(f"🟢 Input - Bio: {bio_text}, Followers: {followers}, Following: {following}, Posts: {posts}")
    
    # Handcrafted rules for obvious fake accounts
    if following > 10 * followers or (posts < 5 and followers < 50):
        return None, {"prediction": "Fake"}
    
    bio_features = vectorizer.transform([bio_text]).toarray()
    ngram_features = count_vectorizer.transform([bio_text]).toarray()
    numeric_features = scaler.transform([[followers, following, posts]])
    sentiment = TextBlob(bio_text).sentiment.polarity
    
    community = communities.get(account_id, -1)
    betweenness_score = betweenness.get(account_id, 0)
    eigenvector_score = eigenvector.get(account_id, 0)
    
    feature_names = ['followers', 'following', 'posts']
    input_df = pd.DataFrame(numeric_features, columns=feature_names)
    bio_df = pd.DataFrame(bio_features, columns=vectorizer.get_feature_names_out())
    ngram_df = pd.DataFrame(ngram_features, columns=count_vectorizer.get_feature_names_out())
    sentiment_df = pd.DataFrame([[sentiment]], columns=['bio_sentiment'])
    graph_df = pd.DataFrame([[community, betweenness_score, eigenvector_score]], 
                            columns=['community', 'betweenness', 'eigenvector'])
    final_features = pd.concat([input_df, bio_df, ngram_df, sentiment_df, graph_df], axis=1)
    print(f"🔹 Final Input Shape: {final_features.shape}")
    return final_features, None

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    final_features, error = preprocess_input(data)
    if error:
        return jsonify(error), 400
    
    prediction = model.predict(final_features)[0]
    return jsonify({"prediction": "Fake" if prediction == 1 else "Real"})

if __name__ == '__main__':
    app.run(debug=True)

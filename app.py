from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
model = SentenceTransformer("all-MiniLM-L6-v2")

def euclidean_distances(array1, large_array):
    array1 = np.array(array1)
    large_array = np.array(large_array)
    distances = np.linalg.norm(large_array - array1, axis=1)/5
    return distances

class SocialSimilar:
    def __init__(self, input_data):
        self.groups = {
            'High': ['General Interests', 'Specific Interests'],
            'Med-High': ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'],
            'Med-Low': ['Tolerance for Political Incorrectness', 'Religious Views', 'Political Opinions'],
            'Low': ['Challenges Milestones']
        }
        # Initially this is being loaded from a local file. In the eventual production version, this will need to be loaded from an external DB
        self.base_df = pd.read_parquet('social_test_embeddings.parquet')
        self.input = input_data
        self.embedded_groups = {}
        # As with the data, this should be loaded from an external storage source
        self.lr_model = pickle.load(open('social_reinforcement.pkl', 'rb'))

    def find_similar(self, top_n: int):
        self.make_input_embedding_dict()
        d1 = model.similarity(np.array(self.embedded_groups['High']), self.base_df['High'].tolist())*4
        d2 = euclidean_distances(self.embedded_groups['Med-High'].tolist(), self.base_df['Med-High'].tolist())*3
        d3 = model.similarity(np.array(self.embedded_groups['Med-Low']), self.base_df['Med-Low'].tolist())*2
        d4 = model.similarity(np.array(self.embedded_groups['Low']), self.base_df['Low'].tolist())
        ps = self.determine_ml_weight()
        product = d1*d2*d3*d4*ps
        product = product.numpy()[0]
        strings_and_relatednesses = [(s, p) for s, p in zip(self.base_df['Full String'], product)]
        strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
        strings, relatednesses = zip(*strings_and_relatednesses)
        return strings[:top_n]

    def make_input_embedding_dict(self):
        for key, value in self.groups.items():
            if key == 'Med-High':
                self.embedded_groups[key] = self.input[value].values
            else:
                grouped_data = str(self.input[value].to_dict())
                embedded_data = model.encode(grouped_data)
                self.embedded_groups[key] = embedded_data.tolist()

    def determine_ml_weight(self):
        # This model returns the likelihood that certain pairs of individuals are compatible
        # It is based on historical feedback from prior user pairings
        combined_strings = [f'Person 1: {self.input}\n\nPerson 2: {p2}' for p2 in self.base_df['Full String']]
        paired_embeddings = model.encode(combined_strings)
        probs = self.lr_model.predict_proba(paired_embeddings)
        predictions = [p[1] for p in probs]
        print(predictions)
        return np.array(predictions)


@app.route('/social-finder', methods=['POST'])
def social_finder():
    req_data = request.get_json()
    general_interests = req_data.get('General Interests', [])
    specific_interests = req_data.get('Specific Interests', [])
    openness = req_data.get('openness', 3)
    conscientiousness = req_data.get('conscientiousness', 3)
    extraversion = req_data.get('extraversion', 3)
    agreeableness = req_data.get('agreeableness', 3)
    neuroticism = req_data.get('neuroticism', 3)
    political_tolerance = req_data.get('Tolerance for Political Incorrectness', '')
    religious_views = req_data.get('Religious Views', 'No specific views')
    political_opinions = req_data.get('Political Opinions', '')
    challenges = req_data.get('Challenges Milestones', '')
    num_matches = req_data.get('Number of Matches', 3)
    input_data = pd.Series({
        'General Interests': general_interests,
        'Specific Interests': specific_interests,
        'openness': openness,
        'conscientiousness': conscientiousness,
        'extraversion': extraversion,
        'agreeableness': agreeableness,
        'neuroticism': neuroticism,
        'Tolerance for Political Incorrectness': political_tolerance,
        'Religious Views': religious_views,
        'Political Opinions': political_opinions,
        'Challenges Milestones': challenges
    })
    soc_sim = SocialSimilar(input_data)
    result = soc_sim.find_similar(num_matches)
    matches = [eval(r) for r in result]

    return jsonify({'Matches': matches})

@app.route('/update-social-ml', methods=['GET'])
def update_social_ml():
    # Load Data
    # Initially this is being loaded from a local file. In the eventual production version, this will need to be loaded from an external DB
    pair_df = pd.read_csv('social_validation_data.csv')
    combined_strings = [f'Person 1: {p1}\n\nPerson 2: {p2}' for p1, p2 in zip(pair_df['Person1'], pair_df['Person2'])]
    combined_embeddings = model.encode(combined_strings)
    comb_df = pd.DataFrame(combined_embeddings)
    comb_df['label'] = pair_df['label']
    input_vals = comb_df.drop('label', axis=1)
    y = comb_df['label']
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(input_vals, y, test_size=0.1, random_state=42)

    # Create a Logistic Regression model
    lr_model = LogisticRegression()

    # Train the model
    lr_model.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = lr_model.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    # Save model
    # As with the data loaded at the start, this model will be saved locally. Ideally, this would be saved externally.
    pickle.dump(lr_model, open('social_reinforcement.pkl', 'wb'))

    return jsonify({'Model Accuracy': accuracy})


if __name__ == '__main__':
    app.run()

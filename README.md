# Social Finder Algorithm

## Overview
This is an algorithm for a potential Social Matchmaking app. The algorithm uses a combination of sentence-transformers 
cosine similarity and euclidean distance measures to find the n most similar matches to a given user's profile. A 
logistic regression model is also utilized to reinforce the matches based on user feedback. Once enough real time 
interactions are collected, the model can be trained such that future engagements will now be weighted by the predicted 
logistic regression compatability score. The full algorithm has been set up as a Flask API. 

It should be noted that the collection of users from which matches are made is currently an artificially generated 
dataset. These are fake users and will need to be replaced with actual user data when such data is available.

There are also two GET API requests that will need to be used from time to time to update the underlying user embeddings
and to update and retrain the logistic regression model.

### 1. API POST Request

### Body
The request body must be a JSON object with the following fields:

| Field                                   | Type             | Description                                           |
|-----------------------------------------|------------------|-------------------------------------------------------|
| `Location`                              | String           | The user's location as a "City, State" string         |
| `Acceptable Distance`                   | Integer          | The distance a person would be willing to travel      |
| `Available Times`                       | Array of Strings | One or more of: "Morning", "Afternoon", "Evening"     |
| `Available Days`                        | Array of Strings | One or more days of the week                          |
| `Frequency`                             | String           | One of: "Weekly", "Bi-weekly", 'Monthly"              |
| `Budget`                                | Integer          | The user's budget for an engagement                   |
| `General Interests`                     | Array of Strings | A list of general interests of the user.              |
| `Specific Interests`                    | Array of Strings | A list of specific interests.                         |
| `openness`                              | Integer (1-5)    | Scale representing openness to experience.            |
| `conscientiousness`                     | Integer (1-5)    | Scale representing conscientiousness.                 |
| `extraversion`                          | Integer (1-5)    | Scale representing extraversion.                      |
| `agreeableness`                         | Integer (1-5)    | Scale representing agreeableness.                     |
| `neuroticism`                           | Integer (1-5)    | Scale representing neuroticism.                       |
| `Tolerance for Political Incorrectness` | String           | Description of tolerance for political incorrectness. |
| `Religious Views`                       | String           | Religious beliefs and preferences.                    |
| `Political Opinions`                    | String           | Political stance or opinions.                         |
| `Challenges Milestones`                 | String           | Recent personal or professional milestones.           |
| `Number of Matches`                     | Integer          | The number of recommendations requested.              |

### Example Request
```json
{
    "Location": "Frisco, Texas",
    "Acceptable Distance": 30,
    "Available Times": ["Afternoon", "Evening"],
    "Available Days": ["Friday", "Saturday", "Sunday"],
    "Frequency": "Weekly",
    "Budget": 100,
    "General Interests": ["AI", "Reading", "Sports", "Traveling"],
    "Specific Interests": ["Golf", "Buffalo Bills", "Disney"],
    "openness": 4,
    "conscientiousness": 3,
    "extraversion": 2,
    "agreeableness": 4,
    "neuroticism": 1,
    "Tolerance for Political Incorrectness": "I have a very limited tolerance for political stupidity. I prefer fact-based news.",
    "Religious Views": "Catholic. But I am not a fan of organized religion",
    "Political Opinions": "Neutral. But I have a very sour opinion of MAGA Republicans",
    "Challenges Milestones": "I recently had a huge advancement and promotion in my job.",
    "Number of Matches": 4
}
```

### Response
#### Success Response
**Status Code:** `200 OK`

**Example Response:**
```json
{
    "Matches": [
        {
            "Challenges Milestones": "Facing challenges in team dynamics",
            "General Interests": "Gaming, Sports, Movies",
            "Political Opinions": "Conservative, believes in personal freedoms",
            "Religious Views": "Agnostic, values moral ethics",
            "Specific Interests": "Strategy games, Team sports",
            "Tolerance for Political Incorrectness": "Moderate, enjoys humor but avoids offensive jokes",
            "agreeableness": 3,
            "conscientiousness": 4,
            "extraversion": 5,
            "neuroticism": 5,
            "openness": 2
        },
        {
            "Challenges Milestones": "Managing the challenges of a new job while retaining personal relationships.",
            "General Interests": "Fashion, Socializing, Culture",
            "Political Opinions": "Progressive: I actively participate in advocacy for various causes.",
            "Religious Views": "Agnostic",
            "Specific Interests": "Styling, Concerts",
            "Tolerance for Political Incorrectness": "High: I think we should have the freedom to express our views openly, regardless of offense.",
            "agreeableness": 4,
            "conscientiousness": 3,
            "extraversion": 5,
            "neuroticism": 5,
            "openness": 2
        }
    ]
}
```

### Error Responses
#### Invalid Request
**Status Code:** `400 Bad Request`

**Example Response:**
```json
{
    "error": "Invalid input format. 'General Interests' must be an array of strings."
}
```

#### Internal Server Error
**Status Code:** `500 Internal Server Error`

**Example Response:**
```json
{
    "error": "An unexpected error occurred. Please try again later."
}
```

### Notes
- Personality trait values must be between 1 and 5.
- The `Number of Matches` field determines how many recommendations will be returned.
- Ensure the request is properly formatted as JSON and contains all required fields.

---

### 2. GET /generate-user-embeddings

Generates user embeddings based on user data.

**Request:**
- Method: GET
- No parameters required.

**Process:**
- Loads user data from `social_testing_data_v2.csv`.
- Embeds selected columns: 'General Interests', 'Specific Interests', 'Tolerance for Political Incorrectness', 'Religious Views', 'Political Opinions', 'Challenges Milestones'.
- Non-embedding columns include 'openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism', 'Location', 'Acceptable Distance', 'Available Times', 'Available Days', 'Frequency', 'Budget'.
- Embeddings are generated using a sentence-transformer model and saved to `social_test_embeddings_v2.parquet`.

**Response:**
- Status Code: 200 OK
- Embeddings saved locally.

---

### 3. GET /update-social-ml

Updates the social machine learning model using pairwise user data.

**Request:**
- Method: GET
- No parameters required.

**Process:**
- Loads data from `social_validation_data.csv`.
- Generates embeddings for pairs of users.
- Trains a Logistic Regression model using the embeddings.
- Evaluates the model and reports accuracy.
- Saves the trained model to `social_reinforcement.pkl`.

**Response:**
- Status Code: 200 OK
- JSON object containing model accuracy:
  ```json
  {"Model Accuracy": 0.92}
  ```

---

## Installation
1. **Clone the repository**:
   ```sh
   git clone https://github.com/Csmith715/SocialFinder.git
   cd SocialFinder
   
2. **Set up the virtual environment:**
   ```sh
   python3 -m venv venv
   source venv/bin/activate

3. **Install the dependencies:**
   ```sh
   pip install -r requirements.txt
   
## Usage
1. **Run the application**
   ```sh
   python app.py
   
2. **POST API calls can then be made to:** http://127.0.0.1:5000/social-finder
3. **GET API requests to retrain the logistic regression model can be made to:** http://127.0.0.1:5000/update-social-ml
4. **GET API requests to update user data can be made to:** http://127.0.0.1:5000/generate-user-embeddings

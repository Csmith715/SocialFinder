# Social Finder Algorithm

## Overview
This is an algorithm for a potential Social Matchmaking app. The algorithm uses a combination of sentence-transformers 
cosine similarity and euclidean distance measures to find the n most similar matches to a given user's profile. The 
algorithm has been set up as a Flask API. 

It should be noted that the collection of users from which matches are made is currently an artificially generated 
dataset. These are fake users and will need to be replaced with actual user data when such data is available.

### Body
The request body must be a JSON object with the following fields:

| Field                                   | Type             | Description                                           |
|-----------------------------------------|------------------|-------------------------------------------------------|
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

## Response
### Success Response
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

#### Unauthorized
**Status Code:** `401 Unauthorized`

**Example Response:**
```json
{
    "error": "Unauthorized. Please provide a valid token."
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

## Notes
- Personality trait values must be between 1 and 5.
- The `Number of Matches` field determines how many recommendations will be returned.
- Ensure the request is properly formatted as JSON and contains all required fields.

## Installation
1. **Clone the repository**:
   ```sh
   git clone https://github.com/Csmith715/socialFinder.git
   cd socialFinder
   
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
import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "bio": "Win a free iPhone! Click the link now!",
    "followers": 1000,
    "following": 500,
    "posts": 10,
    "account_id": 12345
}

response = requests.post(url, json=data)
print("Response:", response.json())

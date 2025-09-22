import requests

url = "http://127.0.0.1:5000/predict"

data = {"features": [60, 12, 1200]}  # Example input

response = requests.post(url, json=data)

print("Response:", response.json())
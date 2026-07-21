import requests
import time

url = 'http://localhost:8000/upload'
files = {'file': ('sample_messy_data.csv', open('sample_data/sample_messy_data.csv', 'rb'))}
response = requests.post(url, files=files)
data = response.json()
session_id = data.get('session_id')
print(f'Session ID: {session_id}')

url = 'http://localhost:8000/analyze'
payload = {'session_id': session_id}
response = requests.post(url, json=payload)
print(response.json())

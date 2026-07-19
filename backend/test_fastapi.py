from fastapi.testclient import TestClient
import json
import asyncio
from app.main import app

client = TestClient(app)

print('--- Uploading File ---')
with open('sample_data/sample_messy_data.csv', 'rb') as f:
    response = client.post('/upload', files={'file': ('sample_messy_data.csv', f)})

print(f'Upload Status: {response.status_code}')
if response.status_code != 200:
    print(response.json())
    exit(1)

session_id = response.json().get('session_id')
print(f'Session ID: {session_id}')

print('--- Triggering Analyze ---')
response = client.post('/analyze', json={'session_id': session_id})
print(f'Analyze Status: {response.status_code}')

try:
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print('Failed to parse JSON:', e)
    print(response.text)

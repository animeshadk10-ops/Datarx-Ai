import requests
import json
import sys

try:
    with open('test_api_log.txt', 'w') as f:
        url = 'http://localhost:8000/upload'
        files = {'file': ('sample_messy_data.csv', open('sample_data/sample_messy_data.csv', 'rb'))}
        response = requests.post(url, files=files)
        f.write(f'Upload response status: {response.status_code}\n')
        data = response.json()
        session_id = data.get('session_id')
        f.write(f'Session ID: {session_id}\n')
        
        if session_id:
            url = 'http://localhost:8000/analyze'
            payload = {'session_id': session_id}
            response = requests.post(url, json=payload)
            f.write(f'Analyze response status: {response.status_code}\n')
            f.write(json.dumps(response.json(), indent=2))
        else:
            f.write('No session ID, stopping.\n')
except Exception as e:
    with open('test_api_err.txt', 'w') as f:
        f.write(str(e))

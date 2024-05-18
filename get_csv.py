import requests
import pandas as pd


# https://hci-analysis.software/api/questions/?format=json
# List of API endpoints
urls = {
    "questions": "https://hci-analysis.software/api/questions/",
    "users": "https://hci-analysis.software/api/users/",
    "prompted": "https://hci-analysis.software/api/prompted/",
    "unprompted": "https://hci-analysis.software/api/unprompted/",
    "noassistance": "https://hci-analysis.software/api/noassistance/",
    "feedback": "https://hci-analysis.software/api/feedback/",
    "feedbackans": "https://hci-analysis.software/api/feedbackans/",
}


# Function to fetch data and save as CSV
def fetch_and_save_as_csv(url, filename):
    response = requests.get(f"{url}?format=json")
    if response.status_code == 200:
        data = response.json()
        df = pd.json_normalize(data)
        df.to_csv(f"data/{filename}.csv", index=False)
    else:
        print(f"Failed to fetch data from {url}. Status code: {response.status_code}")


# Loop through URLs and save data
for key, url in urls.items():
    fetch_and_save_as_csv(url, key)

import requests



url = "http://127.0.0.1:5000/predict"

sample_data = {
    "gender": ["female"],
    "race_ethnicity": ["group B"],
    "parental_level_of_education": ["bachelor's degree"],
    "lunch": ["standard"],
    "test_preparation_course": ["none"],
    "reading_score": [72],
    "writing_score": [74]
}

response = requests.post(url, json=sample_data)
print(response.json())

import requests
import random

# for drift check use endpoint 1439518006479683584
# for skew check use endpoint 7472652657295884288


def post_request(data):
    url = "https://us-central1-aiplatform.googleapis.com/v1/projects/947521528723/locations/us-central1/endpoints/7472652657295884288:predict"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer ya29.a0AfB_byBXVyIyf4AJwMH8F2cacjxpily573ZzWeFcjN2TAmLDxngn1rywrZiqRxROcl0_q5VHERD4bzcbESN2vJj-Jnmb_6zMf_oMDiIvyAcz58Nvb7dlxzwV74PmT9cE_qvER9jN3Pr0hac9nBkKWlbXZcaGCMGle6y4qoVoawMaCgYKAVkSARASFQGOcNnCeurUCcVCXfz7IK3RCzR-xg0178"
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()


def random_value_in_list(list):
    return list[random.randint(0, len(list) - 1)]


jobs = ['blue-collar', 'management', 'technician', 'admin.', 'services', 'retired',
        'self-employed', 'entrepreneur', 'unemployed', 'housemaid', 'student', 'unknown']
martials = ['married', 'divorced', 'single']
educations = ['unknown', 'secondary', 'primary', 'tertiary']
binary = ['no', 'yes']
contacts = ['unknown', 'telephone', 'cellular']
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
          'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
poutcomes = ['success', 'failure', 'unknown', 'other']


data = {
    "instances": [{
        "age": random.randint(18, 100),
        "job": random_value_in_list(jobs),
        "martial": random_value_in_list(martials),
        "education": random_value_in_list(educations),
        "default": random_value_in_list(binary),
        "balance": random.randint(100, 100000),
        "housing": random_value_in_list(binary),
        "loan": random_value_in_list(binary),
        "contact": random_value_in_list(contacts),
        "day": random.randint(1, 31),
        "month": random_value_in_list(months),
        "duration": random.randint(1, 1000),
        "campaign": 1,
        "pdays": -1,
        "previous": 0,
        "poutcome": random_value_in_list(poutcomes)
    }],
}

print(data)
print(post_request(data))

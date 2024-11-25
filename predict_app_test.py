import requests

url = 'http://localhost:9696/survive'

passenger = {
    'Pclass': 2,
    'Age': 28 * 1,
    'Sex': 'female',
    'Embarked' : 'C',
    'SibSp': 3,
    'Parch':1,
    'Fare': 10.5 * 5
}

response = requests.post(url, json=passenger).json()
print(response)




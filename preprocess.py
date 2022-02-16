import json

if __name__ == "__main__":
    adult_attributes = {
        "age": (0, 100),
        "workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov",
                      "Without-pay", "Never-worked"],
        "fnlwgt": (0, 1000000),
        "education": ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th",
                      "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"],
        "education-num": (0, 50),
        "marital-status": ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed",
                           "Married-spouse-absent", "Married-AF-spouse"],
        "occupation": ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty",
                       "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving",
                       "Priv-house-serv", "Protective-serv", "Armed-Forces"],
        "relationship": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
        "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
        "sex": ["Female", "Male"],
        "capital-gain": (0, 50000),
        "capital-loss": (0, 50000),
        "hours-per-week": (0, 24 * 7),
        "native-country": ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
                           "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran",
                           "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal",
                           "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia",
                           "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador",
                           "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"],
        "class": [">50K", "<=50K"]
    }
    adult_examples = []
    with open("adult.data", "r") as train_file:
        lines = train_file.readlines()
        for line in lines[:5]:
            line = line.strip()
            values = line.split(", ")
            adult_examples.append(
                {
                    "age": int(values[0]),
                    "workclass": values[1],
                    "fnlwgt": int(values[2]),
                    "education": values[3],
                    "education-num": int(values[4]),
                    "marital-status": values[5],
                    "occupation": values[6],
                    "relationship": values[7],
                    "race": values[8],
                    "sex": values[9],
                    "capital-gain": int(values[10]),
                    "capital-loss": int(values[11]),
                    "hours-per-week": int(values[12]),
                    "native-country": values[13],
                    "class": values[14]
                }
            )
    print(json.dumps(adult_examples,indent=2))

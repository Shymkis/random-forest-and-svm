import dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def adults():
    features = {
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
                            "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]
    }
    examples = []
    with open("datasets/adult.data", "r") as train_file:
        lines = train_file.readlines()
        for line in lines:
            line = line.strip()
            values = line.split(", ")
            examples.append(
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
    return features, examples[:500]

def blobs(plot=False):
    # dataset.blobs(200,[np.array([1, 1]), np.array([3, 1]), np.array([2, 4])],[np.array([[0.5, 0], [0, 0.5]]), np.array([[0.25, 0], [0, 0.25]]), np.array([[1, 0], [0, 1]])])
    data = pd.read_csv("datasets/blobs.csv", index_col=0)
    examples = data.to_dict(orient="records")
    features = {"0": (0, 1), "1": (0, 1)}
    if plot:
        plt.scatter(data["0"], data["1"], s=5, c=data["class"])
        plt.show()
    return features, examples

def digits():
    data = pd.read_csv("datasets/digits.csv").rename(columns={"label": "class"})
    examples = data.to_dict(orient="records")
    features = {c: (0, 255) for c in data.columns if c.startswith("pixel")}
    return features, examples[:250]

def letters():
    features = {
        "x-box horizontal position of box":     (0, 15),
        "y-box vertical position of box":       (0, 15),
        "width width of box":                   (0, 15),
        "high height of box":                   (0, 15),
        "onpix total # on pixels":              (0, 15),
        "x-bar mean x of on pixels in box":     (0, 15),
        "y-bar mean y of on pixels in box":     (0, 15),
        "x2bar mean x variance":                (0, 15),
        "y2bar mean y variance":                (0, 15),
        "xybar mean x y correlation":           (0, 15),
        "x2ybr mean of x * x * y":              (0, 15),
        "xy2br mean of x * y * y":              (0, 15),
        "x-ege mean edge count left to right":  (0, 15),
        "xegvy correlation of x-ege with y":    (0, 15),
        "y-ege mean edge count bottom to top":  (0, 15),
        "yegvx correlation of y-ege with x":    (0, 15)
    }
    examples = []
    train_file = open("datasets/letter-recognition.data", "r")
    lines = train_file.readlines()
    for line in lines:
        line = line[:-1]
        values = line.replace("\"", "").replace(",", " ").split(" ")
        examples.append({
            "x-box horizontal position of box":     int(values[1]),
            "y-box vertical position of box":       int(values[2]),
            "width width of box":                   int(values[3]),
            "high height of box":                   int(values[4]),
            "onpix total # on pixels":              int(values[5]),
            "x-bar mean x of on pixels in box":     int(values[6]),
            "y-bar mean y of on pixels in box":     int(values[7]),
            "x2bar mean x variance":                int(values[8]),
            "y2bar mean y variance":                int(values[9]),
            "xybar mean x y correlation":           int(values[10]),
            "x2ybr mean of x * x * y":              int(values[11]),
            "xy2br mean of x * y * y":              int(values[12]),
            "x-ege mean edge count left to right":  int(values[13]),
            "xegvy correlation of x-ege with y":    int(values[14]),
            "y-ege mean edge count bottom to top":  int(values[15]),
            "yegvx correlation of y-ege with x":    int(values[16]),
            "class":                                str(values[0])
        })
    return features, examples[:500]

def spirals(plot=False):
    # dataset.spirals(1000, 2, 0.5)
    data = pd.read_csv("datasets/spirals.csv", index_col=0)
    features = {"x": (0, 1), "y": (0, 1)}
    examples = data.to_dict(orient="records")
    if plot:
        plt.scatter(data["x"], data["y"], s=5, c=data["class"])
        plt.show()
    return features, examples

def zoo():
    features = {
        "hair":     [True, False],
        "feathers": [True, False],
        "eggs":     [True, False],
        "milk":     [True, False],
        "airborne": [True, False],
        "aquatic":  [True, False],
        "predator": [True, False],
        "toothed":  [True, False],
        "backbone": [True, False],
        "breathes": [True, False],
        "venemous": [True, False],
        "fins":     [True, False],
        "legs":     [0, 2, 4, 5, 6, 8],
        "tail":     [True, False],
        "domestic": [True, False],
        "catsize":  [True, False]
    }
    examples = []
    train_file = open("datasets/zoo.data", "r")
    lines = train_file.readlines()
    classes = ["mammal", "bird", "reptile", "fish", "amphibian", "insect", "invertebrate"]
    for line in lines:
        line = line[:-1]
        values = line.replace("\"", "").replace(",", " ").split(" ")
        examples.append({
            "hair":     bool(int(values[1])),
            "feathers": bool(int(values[2])),
            "eggs":     bool(int(values[3])),
            "milk":     bool(int(values[4])),
            "airborne": bool(int(values[5])),
            "aquatic":  bool(int(values[6])),
            "predator": bool(int(values[7])),
            "toothed":  bool(int(values[8])),
            "backbone": bool(int(values[9])),
            "breathes": bool(int(values[10])),
            "venemous": bool(int(values[11])),
            "fins":     bool(int(values[12])),
            "legs":     int(values[13]),
            "tail":     bool(int(values[14])),
            "domestic": bool(int(values[15])),
            "catsize":  bool(int(values[16])),
            "class":    classes[int(values[17]) - 1]
        })
    return features, examples

from itertools import *
import math
import random
from tqdm import tqdm

class Decision_Tree:
    def __init__(self, node_label, branches=None):
        self.node_label = node_label
        self.branches = branches if branches is not None else []

    def add_branch(self, branch):
        self.branches.append(branch)

    def decide(self, example):
        for b in self.branches:
            if b["branch_label"] == example[self.node_label]:
                return b["node"].decide(example)
            if isinstance(b["branch_label"], str):
                if "<=" in b["branch_label"] and example[self.node_label] <= float(b["branch_label"][2:]):
                    return b["node"].decide(example)
                if ">" in b["branch_label"] and example[self.node_label] > float(b["branch_label"][1:]):
                    return b["node"].decide(example)
        return self.node_label

    def accuracy(self, examples):
        accuracy = 0
        for ex in examples:
            if ex["class"] == self.decide(ex):
                accuracy += 1
        return accuracy/len(examples)*100

    def __str__(self, level=1):
        ret = str(self.node_label) + "\n"
        for b in self.branches:
            ret += "\t"*level + str(b["branch_label"]) + ": " + b["node"].__str__(level + 1)
        return ret


def plurality_value(examples):
    count = {}
    for ex in examples:
        cl = ex["class"]
        if cl not in count:
            count[cl] = 0
        count[cl] += 1
    return max(count, key=count.get)

def all_equal(examples):
    # iterable = []
    # for e in examples:
    #     iterable.append(e["class"])
    # g = groupby(iterable)
    g = groupby([e["class"] for e in examples])
    return next(g, True) and not next(g, False)

def misclassification(probs):
    return 1 - max(probs.values())

def entropy(probs):
    H = 0
    for P in probs.values():
        H += P*math.log(P, 2)
    return -H

def gini(probs):
    G = 0
    for P in probs.values():
        G += P*(1 - P)
    return G

def impurity(examples, measure):
    totals = {}
    for ex in examples:
        cl = ex["class"]
        if cl not in totals:
            totals[cl] = 0
        totals[cl] += 1

    probs = {}
    for cl, total in totals.items():
        probs[cl] = total/len(examples)

    return measure(probs)

def importance(f, examples, features, measure):
    # Information Gain
    before = impurity(examples, measure)
    after = float("inf")
    split = None
    if isinstance(features[f], tuple):
        # Handle numeric data
        exs = sorted(examples, key=lambda e: e[f])
        mids = []
        for i in range(len(exs) - 1):
            e1 = exs[i]
            e2 = exs[i + 1]
            cl1 = e1["class"]
            cl2 = e2["class"]
            if cl1 != cl2:
                mids.append((e1[f] + e2[f])/2)
        for s in mids:
            low_exs = [e for e in exs if e[f] <= s]
            high_exs = [e for e in exs if e[f] > s]
            aft = (len(low_exs)/len(examples))*impurity(low_exs, measure) + \
                (len(high_exs)/len(examples))*impurity(high_exs, measure)
            if aft < after:
                after = aft
                split = s
    else:
        after = 0
        for v in features[f]:
            exs = [e for e in examples if e[f] == v]
            after += (len(exs)/len(examples))*impurity(exs, measure)
    return before - after, split

def argmax(features, examples, impurity_measure):
    F = None
    V = float("-inf")
    split = None
    for f in features:
        v, s = importance(f, examples, features, impurity_measure)
        if v > V:
            F, V, split = f, v, s
    return F, split

def learn_decision_tree(examples, features, parent_examples, impurity_measure, min_examples=400):
    if not examples:
        return Decision_Tree(plurality_value(parent_examples))
    elif all_equal(examples):
        return Decision_Tree(examples[0]["class"])
    elif not features or len(examples) < min_examples:
        return Decision_Tree(plurality_value(examples))
    else:
        keys = random.sample(list(features), math.floor(math.log2(len(features) + 1)))
        feature_subset = {k: features[k] for k in keys}
        F, split = argmax(feature_subset, examples, impurity_measure)
        tree = Decision_Tree(F)
        if split:
            # Handle numeric data
            split = round(split, 9)
            low_exs = [e for e in examples if e[F] <= split]
            low_subtree = learn_decision_tree(low_exs, features, examples, impurity_measure, min_examples)
            tree.add_branch({"branch_label": "<=" + str(split), "node": low_subtree})

            high_exs = [e for e in examples if e[F] > split]
            high_subtree = learn_decision_tree(high_exs, features, examples, impurity_measure, min_examples)
            tree.add_branch({"branch_label": ">" + str(split), "node": high_subtree})
        else:
            for v in features[F]:
                feats = {k: val for k, val in features.items() if k != F}
                exs = [e for e in examples if e[F] == v]
                subtree = learn_decision_tree(exs, feats, examples, impurity_measure, min_examples)
                tree.add_branch({"branch_label": v, "node": subtree})
        return tree

def n_folds(N, examples):
    random.shuffle(examples)
    folds = []
    for x in range(N):
        train_exs = []
        test_exs = []
        for i, example in enumerate(examples):
            if i % N != x:
                train_exs.append(example)
            else:
                test_exs.append(example)
        folds.append({"train": train_exs, "test": test_exs})
    return folds

if __name__ == "__main__":
    restaurant_features = {
        "Alt":      ["Yes", "No"],
        "Bar":      ["Yes", "No"],
        "Fri":      ["Yes", "No"],
        "Hun":      ["Yes", "No"],
        "Pat":      ["None", "Some", "Full"],
        "Price":    ["$", "$$", "$$$"],
        "Rain":     ["Yes", "No"],
        "Res":      ["Yes", "No"],
        "Type":     ["French", "Thai", "Burger", "Italian"],
        "Est":      ["0-10", "10-30", "30-60", ">60"]
    }
    restaurant_examples = [
        {"Alt": "Yes",  "Bar": "No",    "Fri": "No",    "Hun": "Yes",   "Pat": "Some", "Price": "$$$",  "Rain": "No",   "Res": "Yes",   "Type": "French",   "Est": "0-10",     "class": "Yes"},
        {"Alt": "Yes",  "Bar": "No",    "Fri": "No",    "Hun": "Yes",   "Pat": "Full", "Price": "$",    "Rain": "No",   "Res": "No",    "Type": "Thai",     "Est": "30-60",    "class": "No"},
        {"Alt": "No",   "Bar": "Yes",   "Fri": "No",    "Hun": "No",    "Pat": "Some", "Price": "$",    "Rain": "No",   "Res": "No",    "Type": "Burger",   "Est": "0-10",     "class": "Yes"},
        {"Alt": "Yes",  "Bar": "No",    "Fri": "Yes",   "Hun": "Yes",   "Pat": "Full", "Price": "$",    "Rain": "Yes",  "Res": "Yes",   "Type": "Thai",     "Est": "10-30",    "class": "Yes"},
        {"Alt": "Yes",  "Bar": "No",    "Fri": "Yes",   "Hun": "No",    "Pat": "Full", "Price": "$$$",  "Rain": "No",   "Res": "Yes",   "Type": "French",   "Est": ">60",      "class": "No"},
        {"Alt": "No",   "Bar": "Yes",   "Fri": "No",    "Hun": "Yes",   "Pat": "Some", "Price": "$$",   "Rain": "Yes",  "Res": "Yes",   "Type": "Italian",  "Est": "0-10",     "class": "Yes"},
        {"Alt": "No",   "Bar": "Yes",   "Fri": "No",    "Hun": "No",    "Pat": "None", "Price": "$",    "Rain": "Yes",  "Res": "No",    "Type": "Burger",   "Est": "0-10",     "class": "No"},
        {"Alt": "No",   "Bar": "No",    "Fri": "No",    "Hun": "Yes",   "Pat": "Some", "Price": "$$",   "Rain": "Yes",  "Res": "Yes",   "Type": "Thai",     "Est": "0-10",     "class": "Yes"},
        {"Alt": "No",   "Bar": "Yes",   "Fri": "Yes",   "Hun": "No",    "Pat": "Full", "Price": "$",    "Rain": "Yes",  "Res": "No",    "Type": "Burger",   "Est": ">60",      "class": "No"},
        {"Alt": "Yes",  "Bar": "Yes",   "Fri": "Yes",   "Hun": "Yes",   "Pat": "Full", "Price": "$$$",  "Rain": "No",   "Res": "Yes",   "Type": "Italian",  "Est": "10-30",    "class": "No"},
        {"Alt": "No",   "Bar": "No",    "Fri": "No",    "Hun": "No",    "Pat": "None", "Price": "$",    "Rain": "No",   "Res": "No",    "Type": "Thai",     "Est": "0-10",     "class": "No"},
        {"Alt": "Yes",  "Bar": "Yes",   "Fri": "Yes",   "Hun": "Yes",   "Pat": "Full", "Price": "$",    "Rain": "No",   "Res": "No",    "Type": "Burger",   "Est": "30-60",    "class": "Yes"}
    ]

    proj1_features = {
        "x": (0, 1),
        "y": (0, 1)
    }
    proj1_examples = []
    train_file = open("train-file", "r")
    lines = train_file.readlines()
    for line in lines:
        line = line[:-1]
        values = line.replace("\"", "").replace(",", " ").split(" ")
        proj1_examples.append({
            "x":        float(values[1]),
            "y":        float(values[2]),
            "class":    int(values[0])
        })

    zoo_features = {
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
    zoo_examples = []
    train_file = open("zoo.data", "r")
    lines = train_file.readlines()
    classes = ["mammal", "bird", "reptile", "fish", "amphibian", "insect", "invertebrate"]
    for line in lines:
        line = line[:-1]
        values = line.replace("\"", "").replace(",", " ").split(" ")
        zoo_examples.append({
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

    letter_features = {
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
    letter_examples = []
    train_file = open("letter-recognition.data", "r")
    lines = train_file.readlines()
    for line in lines:
        line = line[:-1]
        values = line.replace("\"", "").replace(",", " ").split(" ")
        letter_examples.append({
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

    # print("Restaurant")
    # N = 5
    # max_depth = 5
    # restaurant_folds = n_folds(N, restaurant_examples)
    # restaurant_train_accuracies = restaurant_test_accuracies = 0
    # for fold in tqdm(restaurant_folds):
    #     restaurant_tree = learn_decision_tree(fold["train"], restaurant_features, [], entropy, 5)
    #     restaurant_train_accuracies += restaurant_tree.accuracy(fold["train"])
    #     restaurant_test_accuracies += restaurant_tree.accuracy(fold["test"])
    # print("Average training set accuracy: " + str(round(restaurant_train_accuracies / N, 2)) + "%")
    # print("Average testing set accuracy: " + str(round(restaurant_test_accuracies / N, 2)) + "%")
    # print()

    # print("Proj1")
    # N = 5
    # max_depth = 5
    # proj1_folds = n_folds(N, proj1_examples)
    # proj1_train_accuracies = proj1_test_accuracies = 0
    # for fold in tqdm(proj1_folds):
    #     proj1_tree = learn_decision_tree(fold["train"], proj1_features, [], entropy, max_depth)
    #     proj1_train_accuracies += proj1_tree.accuracy(fold["train"])
    #     proj1_test_accuracies += proj1_tree.accuracy(fold["test"])
    # print("Average training set accuracy: " + str(round(proj1_train_accuracies / N, 2)) + "%")
    # print("Average testing set accuracy: " + str(round(proj1_test_accuracies / N, 2)) + "%")
    # print()

    N = 5
    num_trees = 50
    proj1_folds = n_folds(N, proj1_examples)
    proj1_train_accuracies = proj1_test_accuracies = 0
    proj1_tree = learn_decision_tree(proj1_folds[0]["train"], proj1_features, [], entropy)
    print(proj1_tree)
    proj1_train_accuracies += proj1_tree.accuracy(proj1_folds[0]["train"])
    proj1_test_accuracies += proj1_tree.accuracy(proj1_folds[0]["test"])
    print("Average training set accuracy: " + str(round(proj1_train_accuracies, 2)) + "%")
    print("Average testing set accuracy: " + str(round(proj1_test_accuracies, 2)) + "%")
    print()

    # print("Zoo")
    # N = 5
    # max_depth = 5
    # zoo_folds = n_folds(N, zoo_examples)
    # zoo_train_accuracies = zoo_test_accuracies = 0
    # for fold in tqdm(zoo_folds):
    #     zoo_tree = learn_decision_tree(fold["train"], zoo_features, [], entropy, max_depth)
    #     zoo_train_accuracies += zoo_tree.accuracy(fold["train"])
    #     zoo_test_accuracies += zoo_tree.accuracy(fold["test"])
    # print("Average training set accuracy: " + str(round(zoo_train_accuracies / N, 2)) + "%")
    # print("Average testing set accuracy: " + str(round(zoo_test_accuracies / N, 2)) + "%")
    # print()

    # print("Letters")
    # N = 5
    # max_depth = 5
    # letter_sample = random.sample(letter_examples, 500)
    # letter_folds = n_folds(N, letter_sample)
    # letter_train_accuracies = letter_test_accuracies = 0
    # for fold in tqdm(letter_folds):
    #     letter_tree = learn_decision_tree(fold["train"], letter_features, [], entropy, max_depth)
    #     letter_train_accuracies += letter_tree.accuracy(fold["train"])
    #     letter_test_accuracies += letter_tree.accuracy(fold["test"])
    # print("Average training set accuracy: " + str(round(letter_train_accuracies / N, 2)) + "%")
    # print("Average testing set accuracy: " + str(round(letter_test_accuracies / N, 2)) + "%")
    # print()

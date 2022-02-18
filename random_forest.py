from doctest import Example
from itertools import *
import math
import matplotlib.pyplot as plt
import numpy as np
import preprocess
import random
from tqdm import tqdm
import time

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
    if len(probs) == 0:
        return 0
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

def importance(f, examples, features, measure, min_prop):
    # Information Gain
    before = impurity(examples, measure)
    after = before
    split = None
    if isinstance(features[f], tuple):
        # Handle numeric data
        exs = sorted(examples, key=lambda e: e[f])

        # Take random subset of data
        if min_prop:
            ends = [random.randint(0, len(exs) - 1), random.randint(0, len(exs))]
            while max(ends) - min(ends) < max(2, min_prop*len(exs)):
                ends = [random.randint(0, len(exs) - 1), random.randint(0, len(exs))]
            exs = exs[min(ends):max(ends)]

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

def argmax(features, examples, impurity_measure, min_prop):
    F = None
    V = float("-inf")
    split = None
    for f in features:
        v, s = importance(f, examples, features, impurity_measure, min_prop)
        if v > V:
            F, V, split = f, v, s
    return F, split

def learn_decision_tree(examples, features, parent_examples, impurity_measure, min_prop=0, min_examples=1, max_depth=float("inf"), depth=0):
    if not examples:
        return Decision_Tree(plurality_value(parent_examples))
    elif all_equal(examples):
        return Decision_Tree(examples[0]["class"])
    elif not features or len(examples) < min_examples or depth == max_depth:
        return Decision_Tree(plurality_value(examples))
    else:
        # Random subset of features at each node
        keys = random.sample(list(features), math.floor(math.log2(len(features) + 1)))
        feature_subset = {k: features[k] for k in keys}

        F, split = argmax(feature_subset, examples, impurity_measure, min_prop)
        tree = Decision_Tree(F)
        if split:
            # Handle numeric data
            split = round(split, 9)
            low_exs = [e for e in examples if e[F] <= split]
            low_subtree = learn_decision_tree(low_exs, features, examples, impurity_measure, min_prop, min_examples, max_depth, depth + 1)
            tree.add_branch({"branch_label": "<=" + str(split), "node": low_subtree})

            high_exs = [e for e in examples if e[F] > split]
            high_subtree = learn_decision_tree(high_exs, features, examples, impurity_measure, min_prop, min_examples, max_depth, depth + 1)
            tree.add_branch({"branch_label": ">" + str(split), "node": high_subtree})
        else:
            for v in features[F]:
                feats = {k: val for k, val in features.items() if k != F}
                exs = [e for e in examples if e[F] == v]
                subtree = learn_decision_tree(exs, feats, examples, impurity_measure, min_prop, min_examples, max_depth, depth + 1)
                tree.add_branch({"branch_label": v, "node": subtree})
        return tree

def random_forest(examples, features, parent_examples, impurity_measure, min_prop=0, num_trees=50, min_examples=1, max_depth=float("Inf")):
    forest = []
    for _ in tqdm(range(num_trees), desc="Generating Forest"):
        forest.append(learn_decision_tree(examples, features, parent_examples, impurity_measure, min_prop, min_examples, max_depth))
    return forest

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

def accuracy(forest, examples):
    accuracy = 0
    for ex in examples:
        votes = {}
        for tree in forest:
            prediction = tree.decide(ex)
            if prediction not in votes:
                votes[prediction] = 0
            votes[prediction] += 1
        if ex["class"] == max(votes, key=votes.get):
            accuracy += 1
    return accuracy/len(examples)*100

if __name__ == "__main__":
    N = 5
    # features, examples = preprocess.adults()
    # features, examples = preprocess.blobs()
    features, examples = preprocess.digits()
    # features, examples = preprocess.letters()
    # features, examples = preprocess.spirals()
    # features, examples = preprocess.zoo()

    folds = n_folds(N, examples)
    train_accuracies = test_accuracies = 0
    start = time.time()
    for fold in folds:
        # forest = random_forest(fold["train"], features, [], misclassification, num_trees=150)
        # forest = random_forest(fold["train"], features, [], gini, num_trees=150)
        forest = random_forest(fold["train"], features, [], entropy, num_trees=150)
        train_accuracies += accuracy(forest, fold["train"])
        test_accuracies += accuracy(forest, fold["test"])
    end = time.time()
    avg_time = round((end - start) / N, 2)
    avg_train = round(train_accuracies / N, 2)
    avg_test = round(test_accuracies / N, 2)
    print("Average training set accuracy: " + str(avg_train) + "%")
    print("Average testing set accuracy: " + str(avg_test) + "%")
    print("Average time elapsed: " + str(avg_time))

    # y1 = []
    # y2 = []
    # y3 = []
    # x = []
    # for i in range(1, 4):
        # y1.append(avg_train)
        # y2.append(avg_test)
        # y3.append(avg_time)
        # x.append(i)

    # plot1 = plt.figure(1)
    # plt.plot(x, y1, label="Average Train Accuracy")
    # plt.plot(x, y2, label="Average Test Accuracy")
    # plt.title("Accuracy vs. Number of Trees, Digits, Tuned")
    # plt.xlabel("Number of Trees")
    # plt.ylabel("Accuracy (%)")
    # plt.legend()

    # plot2 = plt.figure(2)
    # plt.plot(x, y3, label="Average Time Elapsed")
    # plt.title("Time Elapsed vs. Number of Trees, Digits, Tuned")
    # plt.xlabel("Number of Trees")
    # plt.ylabel("Time Elapsed (s)")

    # plt.show()

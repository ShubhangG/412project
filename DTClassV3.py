import math
from collections import Counter

def partition_set_on_column(rows, column, value):
    set1 = [row for row in rows if row[column] >= value]
    set2 = [row for row in rows if not row[column] >= value]
    return (set1, set2)


def uniquecounts(rows):

    results = {}
    for row in rows:
        r = row[len(row) - 1]
        if r not in results:
            results[r] = 0  # add key to dict
        results[r] += 1
    return results


def calculate_entropy(rows):
    results = uniquecounts(rows)
    ent = 0.0

    for k, v in results.items():  # changed
        p = float(v) / len(rows)
        ent -= p * math.log(p, 2)

    return ent


class decisionnode:
    def __init__(self, col=-1, value=None, leaf=False, results=None, ifTrue=None, ifFalse=None):
        self.col = col
        self.value = value
        self.results = results
        self.ifTrueBranch = ifTrue
        self.ifFalseBranch = ifFalse
        self.leaf = leaf


def buildtree(data):
    if len(data) == 0:
        return decisionnode()  # len(rows) is the number of units in a set
    if len(data) < 2: #prune to acoid overfitting
        return decisionnode(leaf=True, results=uniquecounts(data))

    current_score = calculate_entropy(data)

    best_gain = 0.0
    best_set1 = None
    best_set2 = None
    best_col = None
    best_val = None

    column_count = len(data[0]) - 1  # count the # of attributes/columns (exclude the target attribute)
    for col in range(0, column_count):
        column_values = {}
        for row in data:
            column_values[row[col]] = 1

        for value in column_values.keys():
            (set1, set2) = partition_set_on_column(data, col, value)

            # Calculate entropy and gain
            relative_size = float(len(set1)) / len(data)
            set1Entropy = relative_size * calculate_entropy(set1)
            set2Entropy = (1 - relative_size) * calculate_entropy(set2)
            gain = current_score - set1Entropy - set2Entropy

            if gain > best_gain and len(set1) > 0 and len(set2) > 0:  # set must not be empty
                best_set2 = set2
                best_gain = gain
                best_col = col
                best_val = value
                best_set1 = set1

    # Recursive Step
    if best_gain > 0:
        trueBranch = buildtree(best_set1)
        falseBranch = buildtree(best_set2)
        return decisionnode(col=best_col, value=best_val, ifTrue=trueBranch, ifFalse=falseBranch)
    else:
        return decisionnode(leaf=True, results=uniquecounts(data))

def classify(observation, tree):
    if tree.leaf:
        return tree.results
    else:
        if observation[tree.col] >= tree.value:
            return classify(observation, tree.ifTrueBranch)
        else:
            return classify(observation, tree.ifFalseBranch)
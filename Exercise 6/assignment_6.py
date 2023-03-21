import numpy as np
from pathlib import Path
from typing import Tuple
import random

class Node:
    """ Node class used to build the decision tree"""
    def __init__(self):
        self.children = {}
        self.parent = None
        self.attribute = None
        self.value = None

    def classify(self, example):
        if self.value is not None:
            return self.value
        return self.children[example[self.attribute]].classify(example)



def plurality_value(examples: np.ndarray) -> int:
    """Implements the PLURALITY-VALUE (Figure 19.5)"""
    labels = examples[:, -1]
    value, count = 0, 0
    for label in np.unique(labels):
        label_count = np.count_nonzero(labels == label)
        if label_count > count:
            value = label
            count = label_count

    return value


def importance(attributes: np.ndarray, examples: np.ndarray, measure: str) -> int:
    """
    This function should compute the importance of each attribute and choose the one with highest importance,
    A ← argmax a ∈ attributes IMPORTANCE (a, examples) (Figure 19.5)

    Parameters:
        attributes (np.ndarray): The set of attributes from which the attribute with highest importance is to be chosen
        examples (np.ndarray): The set of examples to calculate attribute importance from
        measure (str): Measure is either "random" for calculating random importance, or "information_gain" for
                        caulculating importance from information gain (see Section 19.3.3. on page 679 in the book)

    Returns:
        (int): The index of the attribute chosen as the test

    """

    if (measure == "random"):
        return random.randint(0, len(attributes)-1)
    
    elif measure == "information_gain":
        
        target_col = examples[:, -1]
        # Calculate entropy of target variable
        target_entropy = entropy(target_col)

        max_gain = 0
        max_gain_attr_idx = 0

        # Loop over all attributes
        for i, attr in enumerate(attributes):
            attr_col = examples[:, i]
            # Get unique values and their counts in the attribute column
            attr_vals, attr_counts = np.unique(attr_col, return_counts=True)

            attr_entropy = 0
            # Loop over all unique values in the attribute column
            for j, val in enumerate(attr_vals):
                # Get subset of examples where attribute = value
                sub_examples = examples[attr_col == val]
                # Get target variable column for the subset
                sub_target_col = sub_examples[:, -1]
                # Calculate entropy of the subset's target variable
                sub_target_entropy = entropy(sub_target_col)
                # Calculate weight of the subset in the total examples
                weight = attr_counts[j] / len(examples)
                # Calculate weighted average entropy of the subset's target variable
                attr_entropy += weight * sub_target_entropy

            # Calculate information gain for the attribute
            info_gain = target_entropy - attr_entropy
            # Update max gain and max gain attribute index if current gain is higher
            if info_gain > max_gain:
                max_gain = info_gain
                max_gain_attr_idx = i

        # Return the index of the attribute with the highest information gain
        return max_gain_attr_idx


#A function for calculating entropy. It takes in a col, which is a column of a numpy array, and returns the entropy of that column.
def entropy(col):
    _, counts = np.unique(col, return_counts=True)
    p = counts / len(col)
    return -np.sum(p * np.log2(p))       


def learn_decision_tree(examples: np.ndarray, attributes: np.ndarray, parent_examples: np.ndarray,
                        parent: Node, branch_value: int, measure: str):
    """
    This is the decision tree learning algorithm. The pseudocode for the algorithm can be
    found in Figure 19.5 on Page 678 in the book.

    Parameters:
        examples (np.ndarray): The set data examples to consider at the current node
        attributes (np.ndarray): The set of attributes that can be chosen as the test at the current node
        parent_examples (np.ndarray): The set of examples that were used in constructing the current node’s parent.
                                        If at the root of the tree, parent_examples = None
        parent (Node): The parent node of the current node. If at the root of the tree, parent = None
        branch_value (int): The attribute value corresponding to reaching the current node from its parent.
                        If at the root of the tree, branch_value = None
        measure (str): The measure to use for the Importance-function. measure is either "random" or "information_gain"

    Returns:
        (Node): The subtree with the current node as its root
    """

    # Creates a node and links the node to its parent if the parent exists
    node = Node()
    if parent is not None:
        parent.children[branch_value] = node
        node.parent = parent

    # TODO implement the steps of the pseudocode in Figure 19.5 on page 678

    # If examples is empty
    if (len(examples) == 0):
        node.value = plurality_value(parent_examples)
        return node

    # If all the examples are the same
    if (np.all(examples[:, -1] == examples[0, -1])):
        node.value = examples[0, -1]
        return node
    
    # If attributes is empty
    if (attributes.size == 0):
        node.value = plurality_value(examples)


    else: 
        if (measure == 'random'): 
            chosen_attribute = importance(attributes, examples, 'random')
            node.attribute = chosen_attribute
            for value in np.unique(examples[:, chosen_attribute]):
                examples_subset = examples[examples[:, chosen_attribute] == value]
                learn_decision_tree(examples_subset, attributes, examples, node, value, measure)
        
        if (measure == 'information_gain'):
            chosen_attribute = importance(attributes, examples, 'information_gain')
            node.attribute = chosen_attribute
            for value in np.unique(examples[:, chosen_attribute]):
                examples_subset = examples[examples[:, chosen_attribute] == value]
                learn_decision_tree(examples_subset, attributes, examples, node, value, measure)

    return node



def accuracy(tree: Node, examples: np.ndarray) -> float:
    """ Calculates accuracy of tree on examples """
    correct = 0
    for example in examples:
        pred = tree.classify(example[:-1])
        correct += pred == example[-1]
    return correct / examples.shape[0]


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """ Load the data for the assignment,
    Assumes that the data files is in the same folder as the script"""
    with (Path.cwd() / "train.csv").open("r") as f:
        train = np.genfromtxt(f, delimiter=",", dtype=int)
    with (Path.cwd() / "test.csv").open("r") as f:
        test = np.genfromtxt(f, delimiter=",", dtype=int)
    return train, test




if __name__ == '__main__':

    train, test = load_data()

    # information_gain or random
    measure = "information_gain"

    tree = learn_decision_tree(examples=train,
                    attributes=np.arange(0, train.shape[1] - 1, 1, dtype=int),
                    parent_examples=None,
                    parent=None,
                    branch_value=None,
                    measure=measure)

    print(f"Training Accuracy {accuracy(tree, train)}")
    print(f"Test Accuracy {accuracy(tree, test)}")
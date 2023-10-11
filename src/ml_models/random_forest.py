import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def create_random_forest_model(x_train, y_train):
    clf = RandomForestClassifier(random_state=0)
    clf.fit(x_train, y_train)
    return clf


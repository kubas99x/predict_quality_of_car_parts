import numpy as np
import pandas as pd
from xgboost import XGBClassifier

def create_xgb_model(x_train, y_train):
    clf = XGBClassifier()
    clf.fit(x_train, y_train)
    return clf
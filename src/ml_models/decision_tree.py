import pandas as pd
import numpy as np
import plotly.figure_factory as ff
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def create_decision_tree_model(x_train, y_train):
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(x_train, y_train)

    return clf

def print_decision_tree_stats(clf, x_test, y_test):
    y_pred = clf.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    cm = cm[::-1]
    cm = pd.DataFrame(cm, columns=['pred_0, pred_1'], index=['true_1, true_0'])

    fig = ff.create_annotated_heatmap(
        z=cm.values, 
        x=list(cm.columns), 
        y=list(cm.index),
        colorscale='ice',
        showscale=True,
        reversescale=True)
    
    fig.update_layout(
        width=500,
        height=500,
        title='Confusion Matrix',
        font_size=16)
    
    fig.show()

    print(classification_report(y_test, y_pred))



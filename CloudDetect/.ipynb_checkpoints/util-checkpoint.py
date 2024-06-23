import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score

def read_off(file):
    off_header = file.readline().strip()
    if 'OFF' == off_header:
        n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    else:
        n_verts, n_faces, __ = tuple([int(s) for s in off_header[3:].split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return np.array(verts), np.array(faces)


def visualise(vert):
    fig = plt.figure()
    x,y,z = vert.T
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
                                       mode='markers',
                                      marker = {'size':2,
                                                'line':{'width':2,'color':'DarkSlateGrey'}})])
    fig.show()
    return fig

def get_metrics(y_true, y_pred):
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()
    return {
        'f1_score':f1_score(y_true, y_pred, average='macro',zero_division = 0),
        'precision':precision_score(y_true, y_pred, average='macro',zero_division = 0),
        'recall':recall_score(y_true, y_pred, average='macro',zero_division = 0),
        'balanced_acc':balanced_accuracy_score(y_true, y_pred),
    }
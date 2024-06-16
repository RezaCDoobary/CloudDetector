import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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
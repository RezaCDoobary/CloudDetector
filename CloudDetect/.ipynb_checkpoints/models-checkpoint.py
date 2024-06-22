#motivted by https://github.com/nikitakaraevv/pointnet/blob/master/source/model.py

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Tnet(nn.Module):
   def __init__(self, k=3):
      super().__init__()
      self.k=k
      self.conv1 = nn.Conv1d(k,64,1)
      self.conv2 = nn.Conv1d(64,128,1)
      self.conv3 = nn.Conv1d(128,1024,1)
      self.fc1 = nn.Linear(1024,512)
      self.fc2 = nn.Linear(512,256)
      self.fc3 = nn.Linear(256,k*k)

      self.bn1 = nn.BatchNorm1d(64)
      self.bn2 = nn.BatchNorm1d(128)
      self.bn3 = nn.BatchNorm1d(1024)
      self.bn4 = nn.BatchNorm1d(512)
      self.bn5 = nn.BatchNorm1d(256)
       

   def forward(self, input):
      xb = F.relu(self.bn1(self.conv1(input)))
      xb = F.relu(self.bn2(self.conv2(xb)))
      xb = F.relu(self.bn3(self.conv3(xb)))
      pool = nn.MaxPool1d(xb.size(-1))(xb)
      flat = nn.Flatten(1)(pool)
      xb = F.relu(self.bn4(self.fc1(flat)))
      xb = F.relu(self.bn5(self.fc2(xb)))

      matrix = self.fc3(xb).view(-1,self.k,self.k)# + init
      return matrix


class Transform(nn.Module):
   def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Conv1d(3,64,1)

        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
       

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
       
   def forward(self, input):
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        xb = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)

        xb = F.relu(self.bn1(self.conv1(xb)))

        matrix64x64 = self.feature_transform(xb)
        local_features = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)

        xb = F.relu(self.bn2(self.conv2(local_features)))
        xb = self.bn3(self.conv3(xb))
        xb = nn.MaxPool1d(xb.size(-1))(xb)
        output = nn.Flatten(1)(xb)
        return output, local_features, matrix3x3, matrix64x64

class Segmentation(nn.Module):
   def __init__(self, n_points, classes = 10):
        super().__init__()
        self.n_points = n_points
        self.conv1 = nn.Conv1d(1088,512,1)
        self.conv2 = nn.Conv1d(512, 256,1)
        self.conv3 = nn.Conv1d(256, 128,1)
        self.conv3 = nn.Conv1d(256, 128,1)
        self.conv4 = nn.Conv1d(128, classes, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)
       
   def forward(self, global_features, local_features):
        case = global_features.repeat(self.n_points,1,1)
        case2 = torch.transpose(torch.transpose(case, 0, 1),1,2)
        segment = torch.concat([local_features,case2],1)
        xb = F.relu(self.bn1(self.conv1(segment)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        xb = self.dropout(xb)
        xb = self.conv4(xb)
        res = self.logsoftmax(xb)
        return res

class Classification(nn.Module):
   def __init__(self, n_points, classes = 10):
        super().__init__()
        self.n_points = n_points
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)
        

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)
       
   def forward(self, global_features):
        xb = F.relu(self.bn1(self.fc1(global_features)))
        xb = F.relu(self.bn2(self.dropout(self.fc2(xb))))
        output = self.fc3(xb)
        return self.logsoftmax(output)
       
class PointNet(nn.Module):
    def __init__(self, n_point, classes = 10, segment = False):
        super().__init__()
        self.transform = Transform()
        self.segment = segment
        
        if self.segment:
            self.segmentation = Segmentation(n_point, classes)
        else:
            self.classification = Classification(n_point, classes)
            

    def forward(self, input):
        global_features, local_features, matrix3x3, matrix64x64 = self.transform(input)
        
        if self.segment:
            segmentation_output = self.segmentation(global_features, local_features)
            return segmentation_output
        else:
            classification_output = self.classification(global_features)
            return classification_output


import torch.nn as nn
import torch.nn.functional as F


#       Conv2D(64, (3, 3), padding='valid', input_shape=(28, 28, 1)),
#             Activation('relu'),
#             Conv2D(64, (3, 3)),
#             Activation('relu'),
#             MaxPooling2D(pool_size=(2, 2)),
#             Dropout(0.5),
#             Flatten(),
#             Dense(128),
#             Activation('relu'),
#             Dropout(0.5),
#             Dense(10),
#             Activation('softmax')

class JingyiNet(nn.Module):
    '''
    a.k.a LeNet5
    '''

    def __init__(self):
        '''
         2conv + 3fc = 5 layers
        '''
        super(JingyiNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 5=(10-2)/2+1
        )
        self.dropout2d = nn.Dropout2d(0.5)
        self.fc1 = nn.Sequential(
            nn.Linear(9216, 128),
            nn.ReLU()
        )
        self.dropout1d = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout2d(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc1(x)
        x = self.dropout1d(x)
        x = self.fc2(x)
        return x
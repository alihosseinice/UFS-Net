from pandas import DataFrame
import os

Dataset_Path = 'Dataset/'

# Define 8 class labels
Labels = {
    'Normal': [0, 0, 0],
    'Flame': [1, 0, 0],
    'WhiteSmoke': [0, 1, 0],
    'BlackSmoke': [0, 0, 1],
    'Flame_BlackSmoke': [1, 0, 1],
    'Flame_WhiteSmoke': [1, 1, 0],
    'WhiteSmoke_BlackSmoke': [0, 1, 1],
    'Flame_WhiteSmoke_BlackSmoke': [1, 1, 1],
}

classLabels = ['Flame', 'WhiteSmoke', 'BlackSmoke']

# Create Groundtruth for Dataset Images
DataBase = []
for folder in list(Labels.keys()):  # Read all iamges from 8 folder and Create CSV File
    path = Dataset_Path + folder + '/'
    for filename in os.listdir(path):
        Dataset_item = [folder + '/' + filename]
        for item in Labels[folder]:
            Dataset_item.append(item)
        DataBase.append(Dataset_item)

df = DataFrame(DataBase, columns=['Filename', 'Flame', 'WhiteSmoke', 'BlackSmoke'])
df.to_csv('Dataset/Groundtruth.csv', index=False)

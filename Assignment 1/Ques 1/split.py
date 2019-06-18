import os
import numpy as np
import shutil

root_dir = os.getcwd()


classes = []
for data_file in sorted(os.listdir(root_dir)):
    if data_file.startswith("Class"):
        classes.append(data_file)

for i in classes:
    if (not(os.path.isdir('./train/' + i))):
        os.makedirs('./train/' + i)
        os.makedirs('./test/' + i)

for i in classes:
    imgs = os.listdir('./'+i)
    np.random.shuffle(imgs)
    train_files, test_files = np.split(np.array(imgs), [int(len(imgs)*0.85)])
    train_files = ['./'+i+'/'+ name for name in train_files.tolist()]
    test_files = ['./'+i+'/'+ name for name in test_files.tolist()]
    
    for name in train_files:
        shutil.copy(name, "./train/"+i)
    for name in test_files:
        shutil.copy(name, "./test/"+i)

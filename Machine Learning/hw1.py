from skimage import io,transform
import numpy as np
import os

dataset_dir = '/notebooks/data/youjun/Code'
num_train = 35

dataset_main_folder_list = [name for name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir,name))]
dataset_root = os.path.join(dataset_dir, dataset_main_folder_list[0]) #根目錄 /notebooks/data/youjun/Code/CroppedYale

#排序子目錄
sort_root = os.listdir(dataset_root)
sort_root.sort() 

directories = []
class_names = []

for filename in sort_root:
        path = os.path.join(dataset_root, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)
    

x_train = []
x_test  = []
y_train = []
y_test  = []
            
label_count = 1
for directory in directories:
    count = 0
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if (os.path.splitext(path)[-1]) == '.pgm':
            image = io.imread(path)
            if image.shape != (192,168):
                image = transform.resize(image,(192,168), mode='constant') #resize(192,168)
            if count < num_train:
                x_train.append(image)
                y_train.append(label_count)
            else:
                x_test.append(image)
                y_test.append(label_count)
            count = count + 1
        else:
            continue
    label_count = label_count + 1
    
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)




test_count = 0
correct_count = 0
for test in x_test:
    mini = float("inf")
    train_count = 0
    for train in x_train:
        z = abs(test/100 - train/100)
        z = z*100
        SAD = np.sum(z)
        if SAD < mini:
            mini = SAD
            mini_count = train_count
        train_count = train_count + 1
    if(y_train[mini_count] == y_test[test_count]):
        print(y_train[mini_count],y_test[test_count])
        print(class_names[y_train[mini_count]-1])
        correct_count = correct_count + 1
    test_count = test_count + 1

accuracy = correct_count / test_count
print(accuracy)
            

print(dataset_main_folder_list)
print(dataset_root)
print(os.listdir(dataset_root))
print(sort_root)



print(y_train)
#print(class_names)






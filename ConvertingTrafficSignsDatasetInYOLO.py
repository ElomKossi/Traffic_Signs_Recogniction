import os
import pandas as pd
import  cv2

# Absolute path to the folder with Traffic Signs dataset
full_path_to_ts_dataset = "/Users/joke/Personnel/UTBM/Automne_2020/IN54/Projet/TS_Dataset/FullIJCNN2013/ts_one"

# Lists for categories according to the classes ID's
# Prohibitory category: circular Traffic Signs with white background and red border line
prohibitory = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]

# Danger category: triangular Traffic Signs with white background and red border line
danger = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

# Mandatory category: circular Traffic Signs with blue background
mandatory = [33, 34, 35, 36, 37, 38, 39, 40]

# Other category:
other = [6, 12, 13, 14, 17, 32, 41, 42]

# Loading six columns into Pandas dataFrame
annotation = pd.read_csv(full_path_to_ts_dataset + '/' + 'gt.txt',
    names=['ImageID', 'XMin', 'YMin', 'XMax', 'YMax', 'ClassID'],
    sep=';'
)

print(annotation.head())

# Adding new empty columns to dataFrame to save numbers for YOLO format
annotation['CategoryID'] = ''
annotation['center x'] = ''
annotation['center y'] = ''
annotation['width'] = ''
annotation['height'] = ''

# Getting category's ID according to the class's ID
annotation.loc[annotation['ClassID'].isin(prohibitory), 'CategoryID'] = 0
annotation.loc[annotation['ClassID'].isin(danger), 'CategoryID'] = 1
annotation.loc[annotation['ClassID'].isin(mandatory), 'CategoryID'] = 2
annotation.loc[annotation['ClassID'].isin(other), 'CategoryID'] = 3

# Calculating bounding box's center in x and y for all rows
annotation['center x'] = (annotation['XMax'] + annotation['XMin']) / 2
annotation['center y'] = (annotation['YMax'] + annotation['YMin']) / 2

# Calculating bounding box's width and height for all rows
annotation['width'] = annotation['XMax'] - annotation['XMin']
annotation['height'] = annotation['YMax'] - annotation['YMin']

# Getting Pandas dataFrame that has only needed columns
newAnnotationTab = annotation.loc[:, ['ImageID', 'CategoryID', 'center x',  'center y', 'width', 'height']].copy()
print(newAnnotationTab.head())

# Normalizing YOLO numbers according to the real image width and height
# Changing the current directory to one with images
os.chdir(full_path_to_ts_dataset)

# Fullstop in os.walk('.') means the current directory
for current_dir, dirs, files in os.walk('.'):
    # Going through all files
    for f in files:
        # Reading image and getting its real width and height
        image_ppm = cv2.imread(f)

        # Slicing from tuple only first two elements
        h, w = image_ppm.shape[:2]

        # Slicing only name of the file without extension
        image_name = f[:-4]

        # Getting Pandas dataFrame that has only needed rows
        sub_newAnnotationTab = newAnnotationTab.loc[newAnnotationTab['ImageID'] == f].copy()

        # Normalizing calculated bounding boxes' coordinates
        # according to the real image width and height
        sub_newAnnotationTab['center x'] = sub_newAnnotationTab['center x'] / w
        sub_newAnnotationTab['center y'] = sub_newAnnotationTab['center y'] / h
        sub_newAnnotationTab['width'] = sub_newAnnotationTab['width'] / w
        sub_newAnnotationTab['height'] = sub_newAnnotationTab['height'] / h

        # Getting resulted Pandas dataFrame that has only needed columns
        resulted_frame = sub_newAnnotationTab.loc[:, ['CategoryID', 'center x', 'center y', 'width', 'height']].copy()

        # Checking if there is no any annotations for current image
        if resulted_frame.isnull().values.all():
            # Skipping this image
            continue

        # Preparing path where to save txt file
        path_to_save = full_path_to_ts_dataset + '/' + image_name + '.txt'

        # Saving resulted Pandas dataFrame into txt file
        resulted_frame.to_csv(path_to_save, header=False, index=False, sep=' ')


        # Preparing path where to save jpg image
        path_to_save = full_path_to_ts_dataset + '/' + image_name + '.jpg'

        # Saving image in jpg format by OpenCV function
        # that uses extension to choose format to save with
        cv2.imwrite(path_to_save, image_ppm)
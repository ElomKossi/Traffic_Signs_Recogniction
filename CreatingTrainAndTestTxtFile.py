import os

# Absolute path to the folder with Traffic Signs dataset
full_path_to_images = "/Users/joke/Personnel/UTBM/Automne_2020/IN54/Projet/TS_Dataset/ts"

# Getting list of full paths to downloaded images
# Changing the current directory to one with images
os.chdir(full_path_to_images)

# Defining list to write paths in
paths = []

# Using os.walk for going through all directories and files in them from the current directory
# Fullstop in os.walk('.') means the current directory
for current_dir, dirs, files in os.walk('.'):
    # Going through all files
    for f in files:
        # Checking if filename ends with '.jpg'
        if f.endswith('.jpg'):
            # Preparing path to save into train.txt file
            path_to_save_into_txt_files = full_path_to_images + '/' + f

            # Appending the line into the list
            paths.append(path_to_save_into_txt_files + '\n')

# Slicing first 15% of elements from the list
# to write into the test.txt file
paths_test = paths[:int(len(paths) * 0.15)]

# Deleting from initial list first 15% of elements
paths = paths[int(len(paths) * 0.15):]

# Creating train.txt and test.txt files
# Creating file train.txt and writing 85% of lines in it
with open('train.txt', 'w') as train_txt:
    # Going through all elements of the list
    for e in paths:
        # Writing current path at the end of the file
        train_txt.write(e)

# Creating file test.txt and writing 15% of lines in it
with open('test.txt', 'w') as test_txt:
    # Going through all elements of the list
    for e in paths_test:
        # Writing current path at the end of the file
        test_txt.write(e)
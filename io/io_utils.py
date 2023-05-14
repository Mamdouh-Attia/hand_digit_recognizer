
#defining the input format as follows:
'''
we will receive a folder
named data. Under this directory, you will find N test images. These images files will be
named 1.png, 2.png, 3.png, . . . ., 10.png, 11.png, 12.png, . . . .., 100.png, 101.png,
102.png,. . . .etc.
The hierarchy can be visualized as follows (for the first two test cases):
• \data
– 1.png
– 2.png
– .....
Making sure to process the test images in an increasingly ordered sequence.
'''
import os
import cv2
import numpy as np
#function to read the files in the directory in increasing order they are named as numbers and return them as a list of images
def read_data_folder(data_dir):
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist.")
        os.mkdir(data_dir)
        return []
    filenames = sorted(os.listdir(data_dir), key=lambda x: int(os.path.splitext(x)[0]))
    data = []
    for img_path in filenames:
        full_path = os.path.join(data_dir, img_path)
        img = cv2.imread(full_path, -1)
        if img is None:
            print(f'Image could not be read: {full_path}')
        else:
            data.append(img)
            print(f'Image read: {full_path}')
    return data

# data = read_data_folder('dataset2_large\data')
# print(len(data))


#function to rename the files in the directory to be in increasing order 1.png , 2.png and so on 
def rename_files(data_dir):
    #rename the files in the directory to be in increasing order 1.png , 2.png and so on
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist.")
        os.mkdir(data_dir)
        return []
    filenames = os.listdir(data_dir)
    filenames.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else -1)
    for i, filename in enumerate(filenames):
        os.rename(os.path.join(data_dir, filename), os.path.join(data_dir, f'{i + 1}.png'))
    return filenames


#function to write list of output predictions to a file named results.txt TO DIRECTORY output/
#the output file will look like this:
'''
3
0
2
1
4
2
'''
def write_output_file(output_dir, output):
    if not os.path.exists(output_dir):
        print(f"Directory {output_dir} does not exist.")
        os.mkdir(output_dir)
    with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
        for label in output:
            f.write(f'{label}\n')
    print(f'Output file written to {output_dir}/results.txt')
#testing the function
write_output_file('output', [3, 0, 2, 1, 4, 2])

#function to write the output time to a file named time.txt TO DIRECTORY output/
#the output file will look like this:
'''
30.00
35.205
32.320
40.150
35.121
38.982
26.233
'''
def write_time_file(output_dir, time):
    if not os.path.exists(output_dir):
        print(f"Directory {output_dir} does not exist.")
        os.mkdir(output_dir)
    with open(os.path.join(output_dir, 'time.txt'), 'w') as f:
        for t in time:
            #The running times should be rounded to three decimal places.
            t = round(t, 3)
            f.write(f'{t}\n')
    print(f'Output file written to {output_dir}/time.txt')
#testing the function
write_time_file('output', [30.00, 35.2056, 32.3201, 40.1505, 35.121, 38.982, 26.233])






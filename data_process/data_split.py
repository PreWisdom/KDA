import os
import shutil
from sklearn.model_selection import train_test_split


root_dir = r'E:\CodeRepositories\Hokkaido'
image_dir = os.path.join(root_dir, 'img')
mask_dir = os.path.join(root_dir, 'mask')
train_dir = 'train'
val_dir = 'val'
test_dir = 'test'

os.makedirs(os.path.join(image_dir, train_dir), exist_ok=True)
os.makedirs(os.path.join(image_dir, val_dir), exist_ok=True)
os.makedirs(os.path.join(image_dir, test_dir), exist_ok=True)
os.makedirs(os.path.join(mask_dir, train_dir), exist_ok=True)
os.makedirs(os.path.join(mask_dir, val_dir), exist_ok=True)
os.makedirs(os.path.join(mask_dir, test_dir), exist_ok=True)


def move_files(source_dir, dir, files):
    if not os.path.exists(dir):
        os.makedirs(dir)
    for file_path in files:
        shutil.move(os.path.join(source_dir, file_path), os.path.join(source_dir, dir, os.path.basename(file_path)))


entries = os.listdir(image_dir)
img = [filename for filename in entries if os.path.isfile(os.path.join(image_dir, filename))]

train_img, test_img = train_test_split(img, test_size=0.2, random_state=42)
test_img, val_img = train_test_split(test_img, test_size=0.5, random_state=42)

move_files(image_dir, train_dir, train_img)
move_files(image_dir, test_dir, test_img)
move_files(image_dir, val_dir, val_img)

# 获取image文件夹下的训练集、验证集和测试集的文件列表
image_train_files = os.listdir(os.path.join(image_dir, train_dir))
image_val_files = os.listdir(os.path.join(image_dir, val_dir))
image_test_files = os.listdir(os.path.join(image_dir, test_dir))

# 复制mask文件到对应的文件夹
for image_file in image_train_files:
    mask_file = image_file
    shutil.move(os.path.join(mask_dir, mask_file), os.path.join(mask_dir, train_dir, mask_file))

for image_file in image_val_files:
    mask_file = image_file
    shutil.move(os.path.join(mask_dir, mask_file), os.path.join(mask_dir, val_dir, mask_file))

for image_file in image_test_files:
    mask_file = image_file
    shutil.move(os.path.join(mask_dir, mask_file), os.path.join(mask_dir, test_dir, mask_file))

print('Data Split Done')
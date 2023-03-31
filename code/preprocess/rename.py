import os
import glob

root_dir = "../object_gt/"
dir = root_dir + "dog_images/" 
names = glob.glob(dir + "*.png")
print(names)

for filename in names:
    print(filename.split('/')[-1])
    name, num, temp = filename.split('/')[-1].split('_')

    # num, _ = filename.split('/')[-1].split('.')
    # print(num)
    # num = num.split('_')[0]
    num = int(num)
    print(filename + ' -> ' + dir + '%05d.png' % num)
    os.rename(filename, dir + '%05d.png' % num)

# dir = root_dir + "mask/" 
# names = glob.glob(dir + "*.png")

# for filename in names:
#     print(filename)
#     num, _ = filename.split('/')[-1].split('.')#.split('_')[1]
#     print(num)
#     num = num.split('_')[0]
#     num = int(num)
#     print(filename + ' -> ' + dir + '%05d.png' % num)
#     os.rename(filename, dir + '%05d.png' % num)

# dir = root_dir + "/mask_loss/image/" 
# names = glob.glob(dir + "*.jpg")

# for filename in names:
#     # name, num, temp = filename.split('/')[-1].split['_']C:\Users\Verzo\Desktop\New folder
#     # print(filename.split('/')[-1])
#     # name, num, _ = filename.split('/')[-1].split('_')
#     num, _ = filename.split('/')[-1].split('.')
#     print(num)
#     num = num.split('_')[0]
#     num = int(num)
#     print(filename + ' -> ' + dir + '%05d.jpg' % num)
#     os.rename(filename, dir + '%05d.jpg' % num)

# dir = root_dir + "/mask_loss/mask/" 
# names = glob.glob(dir + "*.png")

# for filename in names:
#     print(filename)
#     num, _ = filename.split('/')[-1].split('.')#.split('_')[1]
#     print(num)
#     num = num.split('_')[0]
#     num = int(num)
#     print(filename + ' -> ' + dir + '%05d.png' % num)
#     os.rename(filename, dir + '%05d.png' % num)


import os
import torch
import numpy as np
import cv2

import moms_apriltag as apt
import numpy as np
import imageio

if __name__ == '__main__':
    family = "tag36h10"
    shape = (6,8)
    filename = "apriltag_target.png"
    size = 50

    tgt = apt.board(shape, family, size)
    imageio.imwrite(filename, tgt)

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def hex_colors_to_rgb_colors(value):
    rgb_colors = np.zeros((len(value), 3), dtype = np.int32)
    for i in range(len(value)):
        rgb_colors[i, :] = hex_to_rgb(value[i])
    return rgb_colors

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

if __name__ == '__main__':

    # shape = (5,3)
    # shape = (10,3)
    # size = 14
    # tag_1 = apt.board(shape, "tag25h9", size).astype(np.uint8)[..., np.newaxis]
    # tag_1 = np.broadcast_to(tag_1, (tag_1.shape[0], tag_1.shape[1], 3))
    # cv2.imwrite("../Checkerboard-A4-25mm-10x7_xjm_100.png", tag_1[0:tag_1.shape[0]//2, :])
    # cv2.imwrite("../Checkerboard-A4-25mm-10x7_xjm_101.png", tag_1[tag_1.shape[0]//2:tag_1.shape[0], :])
    # exit()
    mkdir_ifnotexists('../checkboard/')

    resolution = 1024
    color_num = 6
    patch_size = 140
    image_num = 60

    chessboard = torch.zeros(resolution, resolution, 3)
    channel_color_num = 2
    channel_colors = np.linspace(10, 245, channel_color_num)
    print(channel_colors)
    base_colors = np.zeros((pow(channel_color_num, 3), 3), np.uint8)
    count = 0
    for i in range(channel_color_num):
        for j in range(channel_color_num):
            for k in range(channel_color_num):
                base_colors[count, 0] = channel_colors[i] 
                base_colors[count, 1] = channel_colors[j] 
                base_colors[count, 2] = channel_colors[k]  
                count += 1
    base_colors = base_colors[1:-1, :]
    diff = np.max(base_colors, axis=1) - np.min(base_colors, axis=1)
    valid_mask = (diff > 100) & (np.max(base_colors, axis=1) > 200)
    base_colors = base_colors[valid_mask, :]
    # print(base_colors)
    # exit()
    # color = color.reshape(-1, 3)
    base_colors = np.concatenate([base_colors[0:4, :], base_colors[5:6, :]])
    color_num = color_num - 1
    base_colors = torch.from_numpy(base_colors)
    print(base_colors)
    exit()

    # board_color_0 = base_colors[0].clone()
    # board_color_1 = base_colors[0].clone()
    # board_color_0[1] = 0
    board_color_0 = torch.tensor([0, 90, 120]) * 1.0
    board_color_1 = torch.tensor([100, 0, 120]) * 1.2 
    # board_color_1 = torch.tensor([139, 139, 0]) * 0.7
    print(board_color_0)
    print(board_color_1)

    rand_offset = torch.rand(image_num, 1)
    patch_num_h = chessboard.shape[0] // patch_size 
    patch_num_w = chessboard.shape[1] // patch_size  
    for image_idx in range(image_num):
        offset = 25
        chessboard = torch.ones(resolution, resolution, 3) * 60
        for row in range(chessboard.shape[0] // patch_size):
            for col in range(chessboard.shape[1] // patch_size):
                color_idx = row * (chessboard.shape[1] // patch_size) + col
                if color_idx % 2 == 0:
                    chessboard[row * patch_size + offset : row * patch_size + patch_size + offset, \
                        col * patch_size + offset : col * patch_size + patch_size + offset, :] = board_color_0 
                else:
                    chessboard[row * patch_size + offset : row * patch_size + patch_size + offset, \
                        col * patch_size + offset : col * patch_size + patch_size + offset, :] = board_color_1
        # chessboard_np = chessboard.numpy()
        # chessboard_np = cv2.GaussianBlur(chessboard_np, (15,15), 5, cv2.BORDER_DEFAULT) 
        # chessboard = torch.from_numpy(chessboard_np)

        offset += int((rand_offset[image_idx, 0].item() + 0.3) * 80) 
        xx, yy = np.meshgrid(np.arange(patch_num_w - (3 - 1), dtype=np.float32),
                             np.arange(patch_num_h - 2, dtype=np.float32), indexing='xy')
        grids = np.stack([yy, xx], -1) 
        grids = grids.reshape(-1, 2)
        np.random.shuffle(grids)
        for i in range(0, color_num):
            row = int(grids[i, 0]) + 1
            col = int(grids[i, 1]) + (2 - 1)
            print("sample row, col: {0}, {1}".format(row, col))
            chessboard[row * patch_size + offset:row * patch_size + patch_size + offset, \
                    col * patch_size + offset:col * patch_size + patch_size + offset, :] = base_colors[i] 
        print('------------------')

        shape = (2,11*4)
        size = 9
        tag_1 = apt.board(shape, "tag36h10", size).astype(np.uint8)[..., np.newaxis]
        tag_1 = np.broadcast_to(tag_1, (tag_1.shape[0], tag_1.shape[1], 3))
        tag_span = tag_1.shape[1] // 4
        tag_1_1 = tag_1[:, 0:tag_span]
        tag_1_2 = tag_1[:, tag_span:tag_span*2]
        tag_1_3 = tag_1[:, tag_span*2:tag_span*3]
        tag_1_4 = tag_1[:, tag_span*3:tag_span*4]

        size = 9
        tag_2 = apt.board(shape, "tag36h11", size).astype(np.uint8)[..., np.newaxis]
        tag_2 = np.broadcast_to(tag_2, (tag_2.shape[0], tag_2.shape[1], 3))
        tag_2_1 = tag_2[:, 0:tag_span]
        tag_2_2 = tag_2[:, tag_span:tag_span*2]
        tag_2_3 = tag_2[:, tag_span*2:tag_span*3]
        tag_2_4 = tag_2[:, tag_span*3:tag_span*4]

        pad = np.zeros((tag_1_1.shape[0], chessboard.shape[1], 3))
        # print(pad.shape)
        # print(chessboard.shape)
        chessboard = np.concatenate((pad, chessboard, pad), axis=0)
        if image_idx < image_num // 4: 
            chessboard[0:tag_1_1.shape[0], 0:tag_1_1.shape[1], :] = tag_1_1
            chessboard[-tag_1_2.shape[0]-1:-1, -tag_1_2.shape[1]-1:-1, :] = tag_1_2
        elif image_idx >= image_num // 4 and image_idx < image_num // 2:
            chessboard[0:tag_1_3.shape[0], 0:tag_1_3.shape[1], :] = tag_1_3
            chessboard[-tag_1_4.shape[0]-1:-1, -tag_1_4.shape[1]-1:-1, :] = tag_1_4
        elif image_idx >= image_num // 2 and (image_idx < image_num // 4 * 3):
            chessboard[0:tag_2_1.shape[0], 0:tag_2_1.shape[1], :] = tag_2_1
            chessboard[-tag_2_2.shape[0]-1:-1, -tag_2_2.shape[1]-1:-1, :] = tag_2_2
        else:
            chessboard[0:tag_2_3.shape[0], 0:tag_2_3.shape[1], :] = tag_2_3
            chessboard[-tag_2_4.shape[0]-1:-1, -tag_2_4.shape[1]-1:-1, :] = tag_2_4

        chessboard = cv2.rotate(chessboard, cv2.ROTATE_90_CLOCKWISE) 
        text = str(image_idx)
        cv2.putText(chessboard, text, (100, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (200, 200, 200), 10)

        cv2.imwrite('../checkboard/Checkerboard-A4-25mm-10x7_xjm_' + str(image_idx) + '.png', chessboard)

    # image = torch.zeros(1024, 2048, 3).float().numpy()
    # for i in range(30):
    #     image[i:1024:60, :, 0] = 0.9
    #     image[i:1024:60, :, 1] = 0.9
    #     image[i:1024:60, :, 2] = 0.9
    # for i in range(30):
    #     image[:, i:2048:60, 0] = 0.9
    #     image[:, i:2048:60, 1] = 0.9
    #     image[:, i:2048:60, 2] = 0.9
    # # cv2.imshow("image", image)
    # cv2.imwrite("../test_background.exr", image)
    # image *= 250.0
    # image = image.astype(np.uint8)
    # cv2.imwrite("../test_background.png", image)
    # # cv2.imshow("image", image)
    # # cv2.waitKey(0) 

    # shape = (5,3)
    # size = 14
    # tag_1 = apt.board(shape, "tag36h11", size).astype(np.uint8)[..., np.newaxis]
    # tag_1 = np.broadcast_to(tag_1, (tag_1.shape[0], tag_1.shape[1], 3))
    # cv2.imwrite("../Checkerboard-A4-25mm-10x7_xjm_100.png", tag_1)
    # tag_1 = apt.board(shape, "tag25h9", size).astype(np.uint8)[..., np.newaxis]
    # tag_1 = np.broadcast_to(tag_1, (tag_1.shape[0], tag_1.shape[1], 3))
    # cv2.imwrite("../Checkerboard-A4-25mm-10x7_xjm_101.png", tag_1)
    # exit()
    # exit()

  # for row in range(chessboard.shape[0] // patch_size):
    #     for col in range(chessboard.shape[1] // patch_size):
    #         print("row, col: {0}, {1}".format(row, col))
    #         color_idx = (chessboard.shape[1] // patch_size) + col
    #         print(color_idx)
    #         chessboard[row * patch_size:row * patch_size + patch_size, \
    #             col * patch_size:col * patch_size + patch_size, :] = color[color_idx]

    # print(color)
    # exit()

    # print(color)
    # exit()
    # color = color[torch.randperm(color.shape[0])]
    # color_in_hex = ['#2f4f4f', '#7f0000', '#191970', '#006400', '#9acd32', 
    #                 '#8fbc8f', '#ff0000', '#ff8c00', '#ffd700', '#00ff00',
    #                 '#ba55d3', '#00fa9a', '#e9967a', '#00ffff', '#0000ff',
    #                 '#ff00ff', '#1e90ff', '#dda0dd', '#ff1493', '#87cefa']
    # color = hex_colors_to_rgb_colors(color_in_hex)
    # print(color)

    # # color_num = 10000;
    # # color_idx = torch.rand(color_num, 3)
    # # print((torch.flatten(color_idx) * 24).long())
    # # color = color[(torch.flatten(color_idx) * 24).long()].reshape(color_num, 3)
        # print(tag_1.shape)
        # filename = "../Checkerboard-A4-25mm-10x7_xjm_5.png"
        # imageio.imwrite(filename, tag_1)
        # exit()

        # resolution = 1024
        # height = resolution 
        # width = resolution
        # # chessboard = torch.zeros(height, width, 3)
        # corner_00 = torch.tensor([0, 0, 1])
        # corner_01 = torch.tensor([0, 1, 0])
        # corner_10 = torch.tensor([1, 0, 1])
        # corner_11 = torch.tensor([1, 1, 0])

        # x = np.linspace(0, resolution - 1, resolution)
        # y = x
        # xx, yy = np.meshgrid(x, y)
        # grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel()]).T, dtype=torch.float).long()
        # print(grid_points)
        # input()

        # chessboard = torch.zeros(resolution, resolution, 3)
        # chessboard[grid_points[:, 0], grid_points[:, 1], :] += ((width - 1 - grid_points[:, 0]) * (height - 1 - grid_points[:, 1]) / float(width * height))[:, None] * corner_00[None, :]
        # chessboard[grid_points[:, 0], grid_points[:, 1], :] += ((width - 1 - grid_points[:, 0]) *  (grid_points[:, 1]) / float(width * height))[:, None] * corner_01[None, :]
        # chessboard[grid_points[:, 0], grid_points[:, 1], :] += ((grid_points[:, 0]) *  (height - 1 - grid_points[:, 1]) / float(width * height))[:, None] * corner_10[None, :]
        # chessboard[grid_points[:, 0], grid_points[:, 1], :] += ((grid_points[:, 0]) *  (grid_points[:, 1]) / float(width * height))[:, None] * corner_11[None, :]
        # chessboard = chessboard * 255.0
        # chessboard = chessboard.numpy().astype(np.uint8)

     # for row in range(chessboard.shape[0] // patch_size):
        #     for col in range(chessboard.shape[1] // patch_size):
        #         print("row, col: {0}, {1}".format(row, col))
        #         color_idx = row *(chessboard.shape[1] // patch_size) + col
        #         print(color_idx)
        #         chessboard[row * patch_size:row * patch_size + patch_size, \
        #             col * patch_size:col * patch_size + patch_size, :] = color[color_idx] 
        # final_chessboard = chessboard.clone() 
        # scale = 1.0
        # for col in range(1, chessboard.shape[1]): 
        #     final_chessboard[:, col, :] = chessboard[:, col, :] * scale 
        #     scale *= 0.99
        #     if scale < 0.4:
        #         scale = 1.0
        #     final_chessboard[:, col, :] = final_chessboard[:, col, :] % 255 
        # scale = 1.0
        # for row in range(1, chessboard.shape[0]): 
        #     final_chessboard[row, :, :] = chessboard[row, :, :] * scale 
        #     scale *= 0.99
        #     if scale < 0.4:
        #         scale = 1.0
        #     final_chessboard[row, :, :] = final_chessboard[row, :, :] % 255     
        # chessboard = final_chessboard 

    # scale = 1.005
    # count = 0
    # for col in range(1, chessboard.shape[1]): 
    #     start = chessboard[:, col, :]
    #     end = chessboard[:, col, :] // 30
    #     final_chessboard[:, col, :] = (chessboard.shape[1] - col) / float(chessboard.shape[1]) * start + \
    #         col / float(chessboard.shape[1]) * end
        # final_chessboard[:, col, :] = chessboard[:, col, :] * float(chessboard.shape[1]) / (col * 4)
        # final_chessboard[:, col, :] = final_chessboard[:, col - 1, :] / scale
        # count = count + 1
        # if count == 200:
        #     count = 0
        #     final_chessboard[:, col, :] = chessboard[:, col, :]
    #     final_chessboard[:, col, :] = final_chessboard[:, col, :] % 255 
    # image[patch_center_v:patch_center_v+patch_height, patch_center_u:patch_center_u+patch_width] = color[i, :] 
    # cv2.imwrite('../Checkerboard-A4-25mm-10x7_xjm_4.png', final_chessboard.numpy())

    # chessboard = torch.zeros(3508, 2480, 3)
    # # chessboard[i, j]
    # color_num = 7
    # color = torch.zeros(color_num**3, 3)
    # base_color = torch.arange(1, color_num + 1) * 40 - 30
    # print(base_color)
    # # input()
    # for i in range(color_num):
    #     for j in range(color_num):
    #         for k in range(color_num):
    #             color[i * color_num * color_num + j * color_num + k, 0] = base_color[i]
    #             color[i * color_num * color_num + j * color_num + k, 1] = base_color[j]
    #             color[i * color_num * color_num + j * color_num + k, 2] = base_color[k]
    # # color_num = 10000;
    # # color_idx = torch.rand(color_num, 3)
    # # print((torch.flatten(color_idx) * 24).long())
    # # color = color[(torch.flatten(color_idx) * 24).long()].reshape(color_num, 3)
    # patch_size = 160
    # for row in range(chessboard.shape[0] // patch_size):
    #     for col in range(chessboard.shape[1] // patch_size):
    #         print("row, col: {0}, {1}".format(row, col))
    #         color_idx = row * (chessboard.shape[1] // patch_size) + col
    #         print(color_idx)
    #         chessboard[row * patch_size:row * patch_size + patch_size, \
    #             col * patch_size:col * patch_size + patch_size] = color[color_idx]
    # # image[patch_center_v:patch_center_v+patch_height, patch_center_u:patch_center_u+patch_width] = color[i, :] 
    # cv2.imwrite('../Checkerboard-A4-25mm-10x7_xjm.png', chessboard.numpy())

    # image = cv2.imread('../Checkerboard-A4-25mm-10x7.png')
    # print(image.shape)
    # patch_num = 100
    # base_patch_size = 60
    # patch = torch.ones(patch_num, 2)
    # patch[:, 0] *= base_patch_size
    # patch[:, 1] *= base_patch_size
    # # patch += (torch.rand(patch_num, 2) - 0.5) * (base_patch_size) 
    # patch_center = torch.rand(patch_num, 2)
    # patch_center[:, 0] *= float(image.shape[0])
    # patch_center[:, 1] *= float(image.shape[1])
    # color = torch.arange(1, 25) * 10
    # color_idx = torch.rand(patch_num, 3)
    # print((torch.flatten(color_idx) * 24).long())
    # color = color[(torch.flatten(color_idx) * 24).long()].reshape(patch_num, 3)
    # print(color)
    # for i in range(patch.shape[0]):
    #     patch_center_v = int(patch_center[i, 0])
    #     patch_center_u = int(patch_center[i, 1])
    #     patch_height = int(patch[i, 0])
    #     patch_width = int(patch[i, 1])
    #     image[patch_center_v:patch_center_v+patch_height, patch_center_u:patch_center_u+patch_width] = color[i, :] 
    # cv2.imwrite('../Checkerboard-A4-25mm-10x7_xjm.png', image)


import json
from os import path as osp

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils import data
from torchvision import transforms
import cv2


class VITONDataset(data.Dataset):
    def __init__(self, opt):
        super(VITONDataset, self).__init__()
        self.load_height = opt.load_height
        self.load_width = opt.load_width
        self.semantic_nc = opt.semantic_nc
        self.data_path = osp.join(opt.dataset_dir, opt.dataset_mode)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # load data list
        img_names = []
        c_names = []
        with open(osp.join(opt.dataset_dir, opt.dataset_list), 'r') as f:
            for line in f.readlines():
                img_name, c_name = line.strip().split()
                img_names.append(img_name)
                c_names.append(c_name)

        self.img_names = img_names
        self.c_names = dict()
        self.c_names['unpaired'] = c_names

    def get_parse_agnostic(self, parse, pose_data):
        """
        human parsing label:
            0 background    10 tosor-skin
            1 hat           11 scarf
            2 hair          12 skirt
            3 glove         13 face
            4 sunglasses    14 leftArm
            5 upperclothes  15 rightArm
            6 dress         16 leftLeg
            7 coat          17 rightLeg
            8 socks         18 leftShoe
            9 pants         19 rightShoe
        """
        parse_array = np.array(parse)
        parse_upper = ((parse_array == 5).astype(np.float32) +
                       (parse_array == 6).astype(np.float32) +
                       (parse_array == 7).astype(np.float32))
        parse_neck = (parse_array == 10).astype(np.float32)

        r = 10
        agnostic = parse.copy()

        # mask arms (移除左右手臂)
        for parse_id, pose_ids in [(14, [2, 5, 6, 7]), (15, [5, 2, 3, 4])]:
            mask_arm = Image.new('L', (self.load_width, self.load_height), 'black')
            # 2D 编辑图像函数, 用于编辑上面创建 mask_arm 图片
            mask_arm_draw = ImageDraw.Draw(mask_arm)
            i_prev = pose_ids[0]
            for i in pose_ids[1:]:
                if (pose_data[i_prev, 0] == 0.0 and pose_data[i_prev, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                    continue
                mask_arm_draw.line([tuple(pose_data[j]) for j in [i_prev, i]], 'white', width=r*10)
                pointx, pointy = pose_data[i]
                radius = r*4 if i == pose_ids[-1] else r*15
                mask_arm_draw.ellipse((pointx-radius, pointy-radius, pointx+radius, pointy+radius), 'white', 'white')
                i_prev = i
            parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
            agnostic.paste(0, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

        # mask torso & neck (移除 humanparsing 的上身躯干和脖子)
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_neck * 255), 'L'))

        return agnostic

    def get_img_agnostic(self, img, parse, pose_data):
        parse_array = np.array(parse)
        parse_head = ((parse_array == 4).astype(np.float32) +
                      (parse_array == 13).astype(np.float32))
        parse_lower = ((parse_array == 9).astype(np.float32) +
                       (parse_array == 12).astype(np.float32) +
                       (parse_array == 16).astype(np.float32) +
                       (parse_array == 17).astype(np.float32) +
                       (parse_array == 18).astype(np.float32) +
                       (parse_array == 19).astype(np.float32))

        r = 20
        agnostic = img.copy()
        agnostic_draw = ImageDraw.Draw(agnostic)

        length_a = np.linalg.norm(pose_data[5] - pose_data[2])
        length_b = np.linalg.norm(pose_data[12] - pose_data[9])
        point = (pose_data[9] + pose_data[12]) / 2
        pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
        pose_data[12] = point + (pose_data[12] - point) / length_b * length_a

        # mask arms
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r*10)
        for i in [2, 5]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')
        for i in [3, 4, 6, 7]:
            if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*10)
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')

        # mask torso
        for i in [9, 12]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r*6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r*6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r*12)
        agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')

        # mask neck
        pointx, pointy = pose_data[1]
        agnostic_draw.rectangle((pointx-r*7, pointy-r*7, pointx+r*7, pointy+r*7), 'gray', 'gray')
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))

        return agnostic

    def __getitem__(self, index):
        img_name = self.img_names[index]
        c_name = {}
        c = {}
        cm = {}
        for key in self.c_names:
            c_name[key] = self.c_names[key][index]
            c[key] = Image.open(osp.join(self.data_path, 'cloth', c_name[key])).convert('RGB')
            # BILINEAR Interpolation (h, w) -> (size * h/w, size)
            c[key] = transforms.Resize(self.load_width, interpolation=2)(c[key])
            cm[key] = Image.open(osp.join(self.data_path, 'cloth-mask', c_name[key]))
            # NEAREST Interpolation
            cm[key] = transforms.Resize(self.load_width, interpolation=0)(cm[key])

            c[key] = self.transform(c[key])  # [-1,1]
            cm_array = np.array(cm[key])
            cm_array = (cm_array >= 128).astype(np.float32)
            cm[key] = torch.from_numpy(cm_array)  # [0,1]
            cm[key].unsqueeze_(0)

        # load pose image
        pose_name = img_name.replace('.jpg', '_rendered.png')
        pose_rgb = Image.open(osp.join(self.data_path, 'openpose-img', pose_name))
        pose_rgb = transforms.Resize(self.load_width, interpolation=2)(pose_rgb)
        pose_rgb = self.transform(pose_rgb)  # [-1,1]

        pose_name = img_name.replace('.jpg', '_keypoints.json')
        with open(osp.join(self.data_path, 'openpose-json', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints_2d'] # 数据格式为 x,y,confidence
            pose_data = np.array(pose_data)
            # opense pose 指示图: https://github.com/jiaqianjing/openpose/blob/master/doc/02_output.md#ui-and-visual-output
            pose_data = pose_data.reshape((-1, 3))[:, :2] # [25, 2], 即 25 个姿势关键点的坐标 (x, y)

        # load parsing image
        parse_name = img_name.replace('.jpg', '.png')
        parse = Image.open(osp.join(self.data_path, 'image-parse', parse_name))
        parse = transforms.Resize(self.load_width, interpolation=0)(parse)
        
        parse_agnostic = self.get_parse_agnostic(parse, pose_data)
        parse_agnostic = torch.from_numpy(np.array(parse_agnostic)[np.newaxis,:]).long() # [1, 1024, 768]

        """
        CIHP-PGN classes(https://github.com/jiaqianjing/CIHP_PGN/blob/81b64c3d867745b2fa6bca12848fa9469df88d01/evaluation/test_inst_part_ap.py#L9):
            CLASSES = ['background', 'hat', 'hair', 'glove', 'sunglasses', 'upperclothes',
                       'dress', 'coat', 'socks', 'pants', 'tosor-skin', 'scarf', 'skirt',
                       'face', 'leftArm', 'rightArm', 'leftLeg', 'rightLeg', 'leftShoe', 'rightShoe']
        map VITON-HD human segment label
        """
        labels = {
            0: ['background', [0, 10]],  # background, tosor-skin
            1: ['hair', [1, 2]],         # hat, hair
            2: ['face', [4, 13]],        # sunglasses, face
            3: ['upper', [5, 6, 7]],     # upperclothes, dress, coat
            4: ['bottom', [9, 12]],      # pants, skirt
            5: ['left_arm', [14]],       # leftArm
            6: ['right_arm', [15]],      # rightArm
            7: ['left_leg', [16]],       # leftLeg
            8: ['right_leg', [17]],      # rightLeg
            9: ['left_shoe', [18]],      # leftShoe
            10: ['right_shoe', [19]],    # rightShoe
            11: ['socks', [8]],          # socks
            12: ['noise', [3, 11]]       # glove, scarf
        }
        # [20, 1024, 768]
        parse_agnostic_map = torch.zeros(20, self.load_height, self.load_width, dtype=torch.float)
        # [20, 1024, 768] one-hot
        parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
        # [13, 1024, 768]
        new_parse_agnostic_map = torch.zeros(self.semantic_nc, self.load_height, self.load_width, dtype=torch.float)
        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_agnostic_map[i] += parse_agnostic_map[label]

        # load person image
        img = Image.open(osp.join(self.data_path, 'image', img_name))
        img = transforms.Resize(self.load_width, interpolation=2)(img)
        img_agnostic = self.get_img_agnostic(img, parse, pose_data)
        img = self.transform(img)
        img_agnostic = self.transform(img_agnostic)  # [-1,1]

        result = {
            'img_name': img_name,
            'c_name': c_name,
            'img': img,
            'img_agnostic': img_agnostic,
            'parse_agnostic': new_parse_agnostic_map,
            'pose': pose_rgb,
            'cloth': c,
            'cloth_mask': cm,
        }
        return result

    def __len__(self):
        return len(self.img_names)


class VITONDataLoader:
    def __init__(self, opt, dataset):
        super(VITONDataLoader, self).__init__()

        if opt.shuffle:
            train_sampler = data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.workers, pin_memory=True, drop_last=True, sampler=train_sampler
        )
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch

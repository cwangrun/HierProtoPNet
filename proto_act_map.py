import os
import torch
import torch.utils.data
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
import re
from PIL import Image
import matplotlib.pyplot as plt
from helpers import makedir, get_heatmap
import model


parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')     # "0, 1"
args = parser.parse_args()

Labels_dict = {
    "No Finding": 14,
    "Atelectasis": 0,
    "Cardiomegaly": 1,
    "Effusion": 2,
    "Infiltrate": 3,
    "Mass": 4,
    "Nodule": 5,
    "Pneumonia": 6,
    "Pneumothorax": 7,
    "Consolidation": 8,
    "Edema": 9,
    "Emphysema": 10,
    "Fibrosis": 11,
    "Pleural_Thickening": 12,
    "Hernia": 13,
}

from settings import base_architecture, img_size, prototype_shape, num_classes, prototype_activation_function, \
    add_on_layers_type, root_dir

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)


# construct the model
ppnet = model.build_HierProtoPNet(base_architecture=base_architecture,
                                  pretrained=True, img_size=img_size,
                                  prototype_shape=prototype_shape,
                                  num_classes=num_classes,
                                  prototype_activation_function=prototype_activation_function,
                                  add_on_layers_type=add_on_layers_type)
ppnet = ppnet.cuda()

checkpoint_path = "8low_nopush0.8592.pth"
ppnet.load_state_dict(torch.load(checkpoint_path))
model = torch.nn.DataParallel(ppnet)
model.eval()


transform = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

save_dir_main = './Heatmap/'
makedir(save_dir_main)

# test_class = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltrate', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']
test_class = 'Atelectasis'

gr_path = os.path.join(root_dir, "BBox_List_2017.csv")
df_box = pd.read_csv(gr_path)
n_sample = 0
for img_idx in range(len(df_box)):
    target_name = df_box.iloc[img_idx]['Finding Label']
    image_name = df_box.iloc[img_idx]['Image Index']

    box_x = int(df_box.iloc[img_idx]['Bbox [x'] / 2.0)
    box_y = int(df_box.iloc[img_idx]['y'] / 2.0)
    box_w = int(df_box.iloc[img_idx]['w'] / 2.0)
    box_h = int(df_box.iloc[img_idx]['h]'] / 2.0)

    gt_box = (box_y, box_y + box_h, box_x, box_x + box_w)

    if target_name != test_class:
        continue
    n_sample = n_sample + 1

    img_PIL = Image.open(os.path.join(root_dir, 'data', image_name)).convert('RGB')
    img = transform(img_PIL)

    label_gt = Labels_dict[target_name]

    gt_mask = np.zeros((img.shape[1], img.shape[2]))
    gt_mask[box_y:box_y + box_h, box_x:box_x + box_w] = 1

    with torch.no_grad():

        input = img.unsqueeze(0).cuda()
        target = torch.tensor(label_gt).cuda()

        output_all, min_distances_all, similarities_all = model(input)

        img_np = img.cpu().numpy().transpose(1, 2, 0).astype(np.float32)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        original_img_size1, original_img_size2 = img_np.shape[0], img_np.shape[1]

        num_proto_per_class = model.module.num_prototypes // model.module.num_classes

        proto_index_start = num_proto_per_class * label_gt
        proto_index_end = num_proto_per_class * label_gt + num_proto_per_class

        similarity_maps_high = similarities_all[0][0, proto_index_start:proto_index_end]
        similarity_maps_mid = similarities_all[1][0, proto_index_start:proto_index_end]
        similarity_maps_low = similarities_all[2][0, proto_index_start:proto_index_end]

        low_avg = similarity_maps_low.mean(0)
        high_avg = F.interpolate(similarity_maps_high.unsqueeze(0), low_avg.shape).squeeze(0).mean(0)
        mid_avg = F.interpolate(similarity_maps_mid.unsqueeze(0), low_avg.shape).squeeze(0).mean(0)
        similarity_maps_comb = (high_avg + mid_avg + low_avg) / 3.0
        overlap_map_comb, heatmap_comb = get_heatmap(similarity_maps_comb, gt_box, img_np)
        save_dir = os.path.join(save_dir_main, test_class)
        makedir(save_dir)
        save_path = os.path.join(save_dir, image_name.replace('.png', '_overlap.jpg'))
        plt.imsave(save_path, overlap_map_comb, vmin=0.0, vmax=1.0)




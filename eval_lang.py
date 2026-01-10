import os.path
from tqdm import tqdm
import torch
import cv2
import sys
from argparse import ArgumentParser
from os import makedirs
import numpy as np
from autoencoder.model import Autoencoder
from utils.openclip_encoder import OpenCLIPNetwork


def vis(sem_map, image, clip_model, vis_path, img_ann, model_name, file_name):
    valid_map = clip_model.get_max_across(sem_map)  # 3xkx832x1264
    valid_map = valid_map.squeeze()
    img_ann = list(img_ann)
    for i in range(valid_map.shape[0]):
        scale = 30
        kernel = np.ones((scale, scale)) / (scale ** 2)
        np_relev = valid_map[i].cpu().numpy()
        avg_filtered = cv2.filter2D(np_relev, -1, kernel)
        np_relev = 0.5 * (avg_filtered + np_relev)

        img = np_relev
        img = img - img.min()
        img = img / img.max() * 255
        img = img.astype(np.uint8)
        # img = np.where(img < 130, 0, 255)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        save_path = os.path.join(vis_path, model_name, str(img_ann[i]), file_name)
        cv2.imwrite(save_path, img)
        blend_img = img * 0.5 + image * 0.5
        blend_img = blend_img.astype(np.uint8)
        blend_path = os.path.join(vis_path, model_name, str(img_ann[i])+"_blend", file_name)
        cv2.imwrite(blend_path, blend_img)


def process(lang_file, rgb_file, vis_path, clip_model, model, device, model_name, file_name):
    lang = np.load(lang_file)
    lang = torch.from_numpy(lang).float()
    rgb = cv2.imread(rgb_file)
    lang = lang.unsqueeze(0).to(device=device)
    import time
    time1 = time.time()
    with torch.no_grad():
        lvl, h, w, _ = lang.shape
        restored_feat = model.decode(lang.flatten(0, 2))
        restored_feat = restored_feat.view(lvl, h, w, -1)  # 3x832x1264x512

    clip_model.set_positives(img_ann)
    time2 = time.time()
    print("用时", time2 - time1)

    vis(restored_feat, rgb, clip_model, vis_path, img_ann, model_name, file_name)

def prepare(vis_path, img_ann, model_name):
    path1 = os.path.join(vis_path, model_name)
    makedirs(path1, exist_ok=True)
    img_ann = list(img_ann)
    for i in range(len(img_ann)):
        path2 = os.path.join(path1, str(img_ann[i]))
        makedirs(path2, exist_ok=True)
        path3 = os.path.join(path1, str(img_ann[i])+"_blend")
        makedirs(path3, exist_ok=True)

if __name__ == '__main__':
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--ae_ckpt_path', type=str, default=None)
    parser.add_argument('--vis_path', type=str, default="vis")
    args = parser.parse_args(sys.argv[1:])

    model_path = args.model_path
    model_name = args.model_name

    rgb_path = os.path.join(model_path, "test/ours_3000/renders")
    lang_path = os.path.join(model_path, "test/ours_3000/lang_npy")

    # 可视化用
    # rgb_path = os.path.join(model_path, "video/ours_3000/renders")
    # lang_path = os.path.join(model_path, "video/ours_3000/lang_npy")

    ae_ckpt_path = args.ae_ckpt_path
    vis_path = args.vis_path
    encoder_dims = [256, 128, 64, 32, 3]
    decoder_dims = [16, 32, 64, 128, 256, 256, 512]
    img_ann = {"gray instrument", "flesh tissue", "vascular pattern"}

    makedirs(vis_path, exist_ok=True)
    prepare(vis_path, img_ann, model_name)

    rgb_images = os.listdir(rgb_path)
    lang_npys = os.listdir(lang_path)
    assert len(rgb_images) == len(lang_npys), "the number of images should equal to the number of features"

    # instantiate autoencoder and openclip
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = OpenCLIPNetwork(device)
    checkpoint = torch.load(ae_ckpt_path, map_location=device)
    model = Autoencoder(encoder_dims, decoder_dims).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    for i in tqdm(range(len(rgb_images))):
        name = lang_npys[i].split(".")[-2]
        lang_file = os.path.join(lang_path, name + ".npy")
        rgb_file = os.path.join(rgb_path, name + ".png")
        file_name = rgb_images[i]
        process(lang_file, rgb_file, vis_path, clip_model, model, device, model_name, file_name)

import os, sys, torch
from modules.config import cfg
from modules.utils.logger import setup_logger
from modules.model import build_model
import numpy as np

def main():
    print('main excuter')

def get_model():
    logger = setup_logger(name="ASEN", level=cfg.LOGGER.LEVEL, stream=cfg.LOGGER.STREAM)
    cfg_file = '../asenpp/config/FashionAI/FashionAI.yaml'
    model_path = '../asenpp/runs/pretrained_asen/FashionAI.pth.tar'
    cfg.merge_from_file(cfg_file)
    cfg.freeze()
    device = torch.device(cfg.DEVICE)
    model = build_model(cfg)
    model.to(device)
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        logger.info(f"Loaded checkpoint '{model_path}'")
    else:
        logger.info(f"No checkpoint found at '{model_path}'.")
        sys.exit()
    return model
    
# 获取单张图片特征
def get_single_img_feat(imgTensor, model):
    model.eval()
    gx = imgTensor.cuda()
    a = torch.full((1,), 1).cuda()
    g_feats, attmap = model(gx, a, level='global')
    return g_feats[0], attmap[0]

# 基于查询图像对候选图集排序
def ranking(query, cand_set):
    simmat = np.matmul(query, cand_set.T)
    index = np.argsort(-simmat)
    return index

if __name__ == "__main__":
    main()
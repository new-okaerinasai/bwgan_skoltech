import argparse
import torch
import scipy.misc
import os
from model import Generator
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()
    for param_name in os.listdir(os.path.join(args.path)):
        G = Generator().cuda()
        try:
            G.load_state_dict(torch.load(os.path.join(args.path, param_name, "generator_150000.pth.tar"))["state_dict"])
        except:
            G.load_state_dict(torch.load(os.path.join(args.path, param_name, "generator_100000.pth.tar"))["state_dict"])
        z = torch.randn(4000, 128)
        z = torch.utils.data.DataLoader(z, batch_size=128)
        all_im = []
        for j, batch in enumerate(z):
            imgs = G.forward(batch.cuda())
            imgs += 1
            imgs /= 2
            imgs = imgs.cpu().detach().numpy().transpose((0,2,3,1))
            os.makedirs(os.path.join(args.path, param_name, "generated"), exist_ok=True)
            for i in range(len(imgs)):
                plt.imsave(os.path.join(args.path, param_name, "generated", str(j * 128 + i) + ".jpg"), imgs[i])


import torch
import os
import numpy as np

from utils import LipDatasetTest
from model import LipNet
from preprocessing import TokenConv, wer

weight_dir = "./weights"
best_weight_dir = os.path.join(weight_dir, sorted(os.listdir(weight_dir))[-1])



def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = LipNet().to(device)
    model.load_state_dict(torch.load(best_weight_dir, map_location=device))
    dataset = LipDatasetTest('./dataset', phase='test')
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2)
    ctcdecoder = TokenConv()
    test_wer = []
    model.eval()
    for i, (vid, align, vid_len, align_len) in enumerate(loader):
        vid = vid.to(device)
        align = align.to(device)
        vid_len = vid_len.to(device)
        align_len = align_len.to(device)
        y = model(vid)
        y = torch.argmax(y, dim=2)
        for tru, pre in zip(align.tolist(), y.tolist()):
            true_txt = ctcdecoder.decode(tru)
            true_txt = "".join(true_txt)
            pred_txt = ctcdecoder.ctc_decode(pre)
            this_wer = wer(pred_txt, true_txt)
            print(this_wer)
            test_wer.extend(this_wer)
            print("True: ", true_txt)
            print("Pred: ", pred_txt)
    print("WER: ", np.mean(test_wer))

if __name__ == "__main__":
    main()
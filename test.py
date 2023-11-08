import torch
import os
import numpy as np

from utils import LipDataset
from model import LipNet
from preprocessing import TokenConv, wer

weight_dir = "./weights"
best_weight_dir = os.path.join(weight_dir, sorted(os.listdir(weight_dir))[-1])



def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = LipNet().to(device)
    model.load_state_dict(torch.load(best_weight_dir, map_location=device))
    dataset = LipDataset('./dataset', phase='test')
    loader = torch.utils.data.Dataloader(dataset, batch_size=100, num_workers=16)
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
            true_txt = ctcdecoder.ctc_decode(tru)
            pred_txt = ctcdecoder.ctc_decode(pre)
        
            test_wer.extend(wer(pred_txt, true_txt))
            print("True: ", true_txt)
            print("Pred: ", pred_txt)
    print("WER: ", np.mean(test_wer))

if __name__ == "__main__":
    main()
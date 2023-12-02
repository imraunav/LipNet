import torch
import os
import numpy as np

from utils import LipDatasetTest
from model import LipFormer, LipFormerDecoder
from preprocessing import TokenConv, wer

best_weight_dir = "./weights/lipnet-transformer_1400_wer:0.2440.pt"

max_length = 28
def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = LipFormer().to(device)
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
        target_indexes = [0] + [0] * (max_length - 1)
        for i in range(max_length - 1):
            caption = torch.LongTensor(target_indexes).unsqueeze(0)
            mask = torch.zeros((1, max_length), dtype=torch.bool)
            mask[:, i + 1 :] = True

            with torch.no_grad():
                pred, _ = model(vid[0, i], caption, mask)

            pred_token = pred.argmax(dim=-1)[:, i].item()
            target_indexes[i + 1] = pred_token

            # if pred_token == vocab["<eos>"]:
            #     break
        for tru, pre in zip(align.tolist(), target_indexes.tolist()):
            true_txt = ctcdecoder.decode(tru)
            true_txt = "".join(true_txt)
            pred_txt = ctcdecoder.ctc_decode(pre)
            this_wer = wer(pred_txt, true_txt)
            print(this_wer)
            test_wer.extend(this_wer)
            print("True: ", true_txt)
            print("Pred: ", pred_txt)
    print("WER: ", np.mean(test_wer))
        # target_tokens = [vocab.get_itos()[i] for i in target_indexes]
        # return target_tokens[1:]

if __name__ == "__main__":
    main()
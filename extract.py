import torch
import os
import tqdm
import dnnutil
import model
import dataset
from collections import defaultdict
import numpy as np


def reweight_batch(weights, preds, parts, counts, labels):
    nimg = weights.shape[0]
    nclasses = preds.shape[1]
    combined = torch.zeros((nimg,nclasses)).cuda()
    pi = 0
    for i in range(nimg):
        nparts = int(counts[i].item())
        for n in range(nparts):
            #ignore body which is part 0, so shift all by 1
            part_index = int(parts[pi + n].item()) - 1 
            scaled_row = weights[i, part_index] * preds[pi + n, :]
            combined[i, :] += scaled_row
        pi += nparts
    return combined


def save_full_model_output(net, dataloader, outpath):
    #import pdb; pdb.set_trace()
    logits = []
    labels = []
    net.eval()
    for batch in tqdm.tqdm(dataloader):
        stats, img, label, parts, part_count, partmasks = batch
        with torch.no_grad():
            img, stats, partmasks = dnnutil.network.tocuda([img, stats, partmasks])
            weights, pred = net(stats, img, partmasks)
            pred = reweight_batch(weights, pred, parts, part_count, label)
            pred = pred.cpu().numpy()
            logits.append(pred)
            labels.append(label)

    fnames = [x[0] for x in dataloader.dataset.data]
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0).flatten()

    np.savez_compressed(outpath, pred=logits, label=labels, names=fnames)
    print((logits.argmax(1) == labels).mean())


def save_base_model_output(net, dataloader, outpath):

    logits = []
    labels = []
    net.eval()
    for batch in tqdm.tqdm(dataloader):
        img, label = batch
        with torch.no_grad():
            img = img.cuda()
            pred = net(img) 
            pred = pred.cpu().numpy()
            logits.append(pred)
            labels.append(label)

    fnames = [x[1] for x in dataloader.dataset.data]
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)

    np.savez_compressed(outpath, pred=logits, label=labels, names=fnames)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode')
    parser.add_argument('model', help='Path to model weights')
    parser.add_argument('--blur', action='store_true')
    parser.add_argument('--out', default='model_output', help='Path to output file')
    args = parser.parse_args()

    if args.mode == 'multi':
        datapath = 'papillon/part_crops' + ('_blur' if args.blur else '')
        net = dnnutil.network.load_model(model.MultiwingInvariantResnet, args.model)
        dset = dataset.SegStatsAndWingsDataset(datapath, train=False)
        loader = torch.utils.data.DataLoader(dset, batch_size=80, num_workers=25, collate_fn=dataset.seg_multiwing_collate)
        save_full_model_output(net, loader, args.out)
    elif args.mode == 'base':
        net = dnnutil.network.load_model(model.BFResnet, 'experiments/baseline_weights')
        dset = dataset.FullCropDataset('papillon/full_crops', train=False)
        loader = torch.utils.data.DataLoader(dset, batch_size=50, num_workers=25)
        save_base_model_output(net, loader, args.out)


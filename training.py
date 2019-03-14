import torch
import dnnutil
import numpy as np



def class_accuracy(prediction, label, n_classes):
    mat = np.zeros((n_classes, n_classes))
    preds = prediction.argmax(1)
    ars = [label.detach().cpu().numpy(),preds.detach().cpu().numpy()]
    np.add.at(mat, ars, 1)
    return mat



class WingnetInvariantTrainer(dnnutil.Trainer):

    def __init__(self, net, optim, cls_loss, pair_loss, accuracy_metric, invariant,
                 alpha=.01, epoch_size=None):
        self.net = net
        self.optim = optim
        self.pi_loss = pair_loss
        self.cls_loss = cls_loss
        self.invariant = invariant
        self.measure_accuracy = accuracy_metric
        self.alpha = alpha
        self.epoch_size = epoch_size

    def _loss(self, emb, pred, labels, parts):
        pred_loss = self.cls_loss(pred, labels)
        pi_loss = self.pi_loss(emb, labels, parts)
        loss = pred_loss + self.alpha * pi_loss
        return loss

    def _mask_select(self, emb, pred, part):
        mask = (torch.ones(emb.size()[0], 4, dtype=torch.long) * 
                torch.arange(4, dtype=torch.long)).cuda()
        msk = torch.eq(mask, part.view(-1, 1)).nonzero().t()

        emb = emb[msk[0], msk[1]]
        pred = pred[msk[0], msk[1]]

        return emb, pred

    def train_batch(self, batch):
        self.optim.zero_grad()
        
        #import pdb; pdb.set_trace() 
        if self.invariant:
            imgs, labels, parts = dnnutil.tocuda(batch)
            
            parts = parts % 4

            emb, pred = self.net(imgs)
            emb, pred = self._mask_select(emb, pred, parts)
            loss = self._loss(emb, pred, labels, parts)
 
        else:
       
            imgs, labels, parts = dnnutil.tocuda(batch)
            
            parts = parts % 4

            emb, pred = self.net(imgs)
            emb, pred = self._mask_select(emb, pred, parts)
 
            loss = self.cls_loss(pred, labels)            

        loss.backward()
        self.optim.step()

        loss = loss.item()
        with torch.no_grad():
            accuracy = self.measure_accuracy(pred, labels)
        return loss, accuracy

    def test_batch(self, batch):
        with torch.no_grad():
            imgs, labels, parts = dnnutil.network.tocuda(batch)
            emb, pred = self.net(imgs)
            emb, pred = self._mask_select(emb, pred, parts)
            loss = self._loss(emb, pred, labels, parts).item()
            accuracy = self.measure_accuracy(pred, labels)
        return loss, accuracy


class SegMultiwingTrainer(dnnutil.Trainer):

    def __init__(self, net, optim, cls_loss, accuracy_metric):
        super(SegMultiwingTrainer, self).__init__(net, optim, cls_loss, accuracy_metric)

    def _mask_select(self, pred, part):
        mask = (torch.ones(pred.size()[0], 4, dtype=torch.long) * 
                torch.arange(4, dtype=torch.long)).cuda()
        msk = torch.eq(mask, part.view(-1, 1)).nonzero().t()

        pred = pred[msk[0], msk[1]]

        return pred

    def reweight_each_batch(self, weights, perpart_preds, part_ids, part_counts, labels, debug=False):
        nimg = weights.shape[0]
        nclasses = perpart_preds.shape[1]
        combined = torch.zeros( (nimg,nclasses) ).cuda()
        pi = 0
        for i in range(nimg):
            nparts = int(part_counts[i].item())
            for n in range(nparts):
                #ignore body which is part 0, so shift all by 1, actually don't do that -Greg
                part_index  = int(part_ids[pi+n].item()) #    - 1 
                scaled_row = weights[i,part_index] * perpart_preds[pi+n,:]
                combined[i,:] += scaled_row
            pi += nparts
        return combined

    def train_batch(self, batch, debug=False, nclasses=150):
        self.optim.zero_grad()
        
        batch = dnnutil.tocuda(batch)
        stats, parts, labels, partids, partcnts, vmasks = batch
        

        partids_8 = partids
        partids = partids % 4


        #import pdb; pdb.set_trace()
        weights, perpart_preds = self.net(stats, parts, vmasks)
        perpart_preds = self._mask_select(perpart_preds, partids)
        labels_flat = labels.view(-1)

        # Construct weighted sum of perpart_preds
        combined_pred = self.reweight_each_batch(
            weights, perpart_preds, partids_8, partcnts, labels_flat)
        loss = self.loss_fn(combined_pred, labels_flat)

        loss.backward()
        self.optim.step()

        loss = loss.item()
        with torch.no_grad():
            accuracy = self.measure_accuracy(combined_pred, labels_flat)
            if nclasses != 0:
                class_acc = class_accuracy(combined_pred, labels_flat,nclasses)
        if nclasses != 0:
            return loss, accuracy, class_acc
        else:
            return loss, accuracy

    def test_batch(self, batch, nclasses=150):
        with torch.no_grad():
            stats, parts, labels, partids, partcnts, vmasks = dnnutil.network.tocuda(batch)
            weights, perpart_preds = self.net(stats, parts, vmasks)

            #import pdb; pdb.set_trace()

            partids_8 = partids
            partids = partids % 4


            perpart_preds = self._mask_select(perpart_preds, partids)
            labels_flat = labels.view(labels.numel())

            combined_pred = self.reweight_each_batch(
                weights, perpart_preds, partids_8, partcnts, labels_flat)

            loss = self.loss_fn(combined_pred, labels_flat).item()
            accuracy = self.measure_accuracy(combined_pred, labels_flat)
            if nclasses != 0:
                class_acc = class_accuracy(combined_pred, labels_flat,nclasses)
                return loss, accuracy, class_acc
            else:
                return loss, accuracy



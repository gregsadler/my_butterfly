

class SegMultiwingTrainer(Trainer):

    def __init__(self, net, optim, cls_loss, accuracy_metric):
        self.net = net
        self.optim = optim
        self.cls_loss = cls_loss
        self.measure_accuracy = accuracy_metric

    def reweight_each_batch(self, weights, perpart_preds, part_ids, part_counts, labels, debug=False):
        nimg = weights.shape[0]
        nclasses = perpart_preds.shape[1]
        combined = torch.zeros( (nimg,nclasses) ).cuda()
        pi = 0
        for i in range(nimg):
            nparts = int(part_counts[i].item())
            for n in range(nparts):
                #ignore body which is part 0, so shift all by 1
                part_index  = int(part_ids[pi+n].item()) - 1 
                scaled_row = weights[i,part_index] * perpart_preds[pi+n,:]
                combined[i,:] += scaled_row
            pi += nparts
        return combined

    def train_batch(self, batch, debug=False):
        self.optim.zero_grad()
        
        batch = dnnutil.network.tocuda(batch)
        stats, parts, labels, partids, partcnts, vmasks = batch
        stats.requires_grad_()

        weights, perpart_preds = self.net(stats, parts, vmasks)
        labels_flat = labels.view(-1)

        # Construct weighted sum of perpart_preds
        combined_pred = self.reweight_each_batch(
            weights, perpart_preds, partids, partcnts, labels_flat)
        loss = self.cls_loss(combined_pred, labels_flat)
        loss.backward()
        self.optim.step()

        loss = pred_loss.item()
        with torch.no_grad():
            accuracy = self.measure_accuracy(combined_pred, labels_flat)
        return loss, accuracy

    def test_batch(self, batch):
        with torch.no_grad():
            stats, parts, labels, partids, partcnts, vmasks = dnnutil.network.tocuda(batch)
            weights, perpart_preds = self.net(stats, parts, vmasks)
            labels_flat = labels.view(labels.numel())

            combined_pred = self.reweight_each_batch(
                weights, perpart_preds, partids, partcnts, labels_flat)

            loss = self.cls_loss(combined_pred, labels_flat).item()
            accuracy = self.measure_accuracy(combined_pred, labels_flat)
        return loss, accuracy


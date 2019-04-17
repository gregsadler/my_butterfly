import torch
import numpy as np
import dataset
import model
import training
import time
from pathlib import Path
from types import SimpleNamespace
import dnnutil





class AllPairContrastLoss(torch.nn.Module):

    def __init__(self, m=1.0):
        # we want the margin to be somewhat small. We need the contrast loss
        # in order to avoid mode collapse, but we care most about pushing 
        # same-class representations close together
        super(AllPairContrastLoss, self).__init__()
        self.m = m
    
    def forward(self, x, labels, parts):
        pairwise_dist_sq = torch.mm(x, x.t())
        squared_norm = pairwise_dist_sq.diag()
        pairwise_dist_sq = (
            squared_norm.view(1, -1) + 
            squared_norm.view(-1, 1) - 
            2 * pairwise_dist_sq)
        pairwise_dist_sq.clamp_(min=0.0)

        same_label = labels.view(-1, 1).eq(labels)
        same_part = parts.view(-1, 1).eq(parts)

        ## only taking top half of matrix, so not duplicating every element in symmetrical matrix
        same_label_flat  = same_label[torch.triu(torch.ones(x.shape[0],x.shape[0])).eq(1)]    
        same_part_flat  = same_part[torch.triu(torch.ones(x.shape[0],x.shape[0])).eq(1)]    
        
        pairwise_dist_sq_flat = pairwise_dist_sq[torch.triu(torch.ones(x.shape[0],x.shape[0])).eq(1)]

        pairs_same = pairwise_dist_sq_flat[same_label_flat * same_part_flat]
        pairs_diff = pairwise_dist_sq_flat[(1-same_label_flat)*same_part_flat]

        n_same = pairs_same.size(0)

        perm = torch.randperm(pairs_diff.size(0))
        idx = perm[:n_same]
        pairs_diff_new = pairs_diff[idx]

        loss_pos = pairs_same.mean()
        loss_neg = torch.clamp(self.m - pairs_diff_new.sqrt(), min=0).pow(2).mean()
        loss = loss_pos + loss_neg
        return loss



def accuracy(prediction, label):
    acc = torch.mean(torch.eq(prediction.argmax(1), label).float()).item()
    return acc


def setup_vanilla_wings(args):

    root = '/multiview/datasets/papillon/part_crops_new'
    train_data = dataset.WingDataset(root, train=True)
    test_data = dataset.WingDataset(root, train=False)

    kwargs = {
        'batch_size': args.batch_size,
        'num_workers': min(16, args.batch_size // 2),
    }
    DataLoader = torch.utils.data.DataLoader
    train_load = DataLoader(train_data, shuffle=True, **kwargs)
    test_load = DataLoader(test_data, shuffle=False, **kwargs)

    net = dnnutil.load_model(model.BFResnet, args.model)
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    trainer = dnnutil.ClassifierTrainer(net, opt, loss_fn, accuracy)

    state = SimpleNamespace(
        train_load=train_load,
        test_load=test_load,
        net=net,
        loss_fn=loss_fn,
        opt=opt,
        trainer=trainer)

    return state


def setup_wingnet(args):

    #root = '/multiview/datasets/papillon/part_crops_gen'
    root = '/multiview/datasets/papillon/part_crops_new'

    largs = dict(
        batch_size=args.batch_size,
        num_workers=min(24, args.batch_size // 2),
    )

    kwargs = dict()
    invariant = args.invariant
    if invariant:
        kwargs.update(num_trans=args.num_trans )##return part ids with class labels and images
        dataset_class = dataset.InvariantWingDataset
        largs.update(collate_fn=dataset.multi_collate)
    else:
        kwargs.update(parts=True)
        dataset_class = dataset.WingDataset

    train_data = dataset_class(root, True, **kwargs)
    test_data = dataset_class(root, False, **kwargs)

    n = len(train_data)
    m = len(test_data)
    tr_samp = dataset.RandomSubsetSampler(n, n // 6)
    te_samp = dataset.RandomSubsetSampler(m, m // 6)
    DataLoader = torch.utils.data.DataLoader
    train_load = DataLoader(train_data, sampler=tr_samp, **largs)
    test_load = DataLoader(test_data, sampler=te_samp, **largs)

    nargs = dict(branches=4, invar=invariant,  model='resnet18')
    if args.init_pretrained:
        net = dnnutil.load_model(model.Wingnet, args.init_pretrained, **nargs)
        net.resnet = dnnutil.load_model(model.ResnetBase, args.init_pretrained)
    else:
        net = dnnutil.network.load_model(model.Wingnet, args.init_pretrained, **nargs)

    if args.freeze:
        try:
            net.freeze_base_()
            params = net.head.parameters()
        except AttributeError:
            net.module.freeze_base_()
            params = net.module.head.parameters()
    else:
        params = net.parameters()

    loss = torch.nn.CrossEntropyLoss()
    siam_loss = AllPairContrastLoss()
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    
    trainer = training.WingnetInvariantTrainer(
        net, opt, loss, siam_loss, accuracy, invariant, alpha=args.alpha)

    state = SimpleNamespace(
        train_load=train_load,
        test_load=test_load,
        net=net,
        opt=opt,
        trainer=trainer)

    return state


def setup_multiwing_weighting(args, use50=False):

    root = '/multiview/datasets/papillon/part_crops_new'
    train_data = dataset.SegStatsAndWingsDataset(root, train=True)
    test_data = dataset.SegStatsAndWingsDataset(root, train=False)

    collate = dataset.seg_multiwing_collate
    kwargs = {
        'batch_size': args.batch_size,
        'num_workers': 16,
        'collate_fn': collate,
    }
    DataLoader = torch.utils.data.DataLoader
    train_load = DataLoader(train_data, shuffle=True, **kwargs)
    test_load = DataLoader(test_data, shuffle=False, **kwargs)
    
    special = args.special ## special modification, use resnet base on each wing, no invariant layers
#    import pdb; pdb.set_trace()
    if args.init_pretrained:
        net = dnnutil.network.load_model(model.WingnetA, args.init_pretrained, special=special)
        if special:
            net.basenet = dnnutil.network.load_model(model.BFResnet, args.init_pretrained)
        else:
            net.basenet = dnnutil.network.load_model(model.Wingnet,
                        args.init_pretrained, branches=4)
    else:
        net = dnnutil.network.load_model(model.WingnetA, args.model)

    params = net.parameters()
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    trainer = training.SegMultiwingTrainer(net, opt, loss_fn, accuracy, special=special)

    state = SimpleNamespace(
        train_load=train_load,
        test_load=test_load,
        net=net,
        loss_fn=loss_fn,
        opt=opt,
        trainer=trainer)

    return state


def get_setup(args):
    method = args.method
    if method == 'base':
        state = setup_vanilla_wings(args)
    elif method == 'invar':
        state = setup_wingnet(args )
    elif method == 'multi':
        state = setup_multiwing_weighting(args)
    else:
        raise NotImplementedError(f'Method {method} is not yet supported')

    return state


def main():
    parser = dnnutil.basic_parser(lr=1e-4, batch_size=24)
    parser.add_argument('method', choices=['base', 'invar', 'multi',],
        help='Which stage to train')
    parser.add_argument('--alpha', type=float, default=0.01,
        help='Loss tradeoff (for invar model)')
    parser.add_argument('--init-pretrained', metavar='WEIGHTS', default='',
        help='Path to the weights of a model from which the model being '
             'trained is derived')
    parser.add_argument('--num-trans', type=int, default=4,
        help='Number of transforms to perform for the transform invariant layer')
    parser.add_argument('--freeze', action='store_true',
        help='Freeze the weights of the base network')
    parser.add_argument('--invariant', action='store_true',
        help='Train Invariant network with contrast loss')
    parser.add_argument('--drop', action='store_true',
        help='Drop Learning rate by one order of magnitude after 25 epochs')
    parser.add_argument('--special', action='store_true',
        help='Use resnet, without wing branches')
    args = parser.parse_args()


    manager = dnnutil.Manager('runs', args.rid)
    manager.set_description(args.note)
    args = manager.load_state(args, restore_lr=False)

    state = get_setup(args)
    train_load, test_load = state.train_load, state.test_load
    net = state.net
    optim = state.opt

    trainer = state.trainer
    logger = dnnutil.TextLog(manager.run_dir / 'log.txt')
    checkpointer = dnnutil.Checkpointer(manager.run_dir, save_multi=True, period=5)

    logger.text(str(args))

    with manager.run_dir.joinpath('.lr').open('w') as f:
        f.write(str(args.lr))
    
    if args.method == 'multi':
        n_classes = 150
    else:
        n_classes = 0

    for e in range(args.start, args.start + args.epochs):
        

        with manager.run_dir.joinpath('.lr').open('r') as f:
            lr = float(f.read().strip())
            if args.drop and e >= 25:
                optim.param_groups[-1]['lr'] = lr*.1
            else:
                optim.param_groups[-1]['lr'] = lr
        
        start = time.time()
        if n_classes != 0:
            train_loss, train_acc, train_class_acc = trainer.train(train_load, e, nclasses = n_classes)
            test_loss, test_acc, test_class_acc  = trainer.eval(test_load, e,nclasses = n_classes)
            np.savez("class_acc/multi/pmatrix_{}".format(str(e)),test = test_class_acc, train=train_class_acc)        
        else:
            train_loss, train_acc = trainer.train(train_load, e, nclasses = n_classes)
            test_loss, test_acc   = trainer.eval(test_load, e,nclasses = n_classes)
        #import pdb; pdb.set_trace()
        t = time.time() - start
        lr = optim.param_groups[-1]['lr']
        logger.log(e, t, train_loss, train_acc, test_loss, test_acc, lr)
        checkpointer.checkpoint(net, test_loss, e)


if __name__ == '__main__':
    main()


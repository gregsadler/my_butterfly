import torch
import torchvision
import dnnutil
import numpy as np
import time
import dataset


def get_data(args):
    root = '/multiview/datasets/papillon/full_crops'
    dargs = dict(batch_size=args.batch_size, num_workers=8)
    train_data = dataset.ButterflyDataset(root, True, size=args.img_size)
    test_data = dataset.ButterflyDataset(root, False, size=args.img_size)
    train_load = torch.utils.data.DataLoader(train_data, shuffle=True, **dargs)
    test_load = torch.utils.data.DataLoader(test_data, shuffle=True, **dargs)

    return train_load, test_load


def get_net(args,checkpoint, nclasses = 150):
    #import pdb; pdb.set_trace()
    net_type = getattr(torchvision.models, args.network)
    net = net_type()
    net.fc = torch.nn.Linear(2048, nclasses)
    cuda = torch.cuda.is_available() 
    if cuda:
        net = net.cuda()
    if checkpoint:
        if cuda:
            params = torch.load(checkpoint)
        else:
            params = torch.load(checkpoint, map_location=lambda s, l:s)
        net.load_state_dict(params, strict =False)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    return net


def accuracy(prediction, label):
    return torch.mean(torch.eq(prediction.argmax(1), label).float()).item()


def main():
    parser = dnnutil.basic_parser(lr=1e-4, batch_size=32)
    parser.add_argument('--network', default='resnet18',
                        help='Network model to train')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Size of the images input to the network')
    args = parser.parse_args()

    manager = dnnutil.Manager('baseline_runs', args.rid)
    manager.set_description(args.note)
    args = manager.load_state(args, restore_lr=False)

    train_load, test_load = get_data(args)
    net = get_net(args, args.model)
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    trainer = dnnutil.ClassifierTrainer(net, optim, loss_fn, accuracy)
    logger = dnnutil.TextLog(manager.run_dir / 'log.txt')
    checkpointer = dnnutil.Checkpointer(manager.run_dir)
    
    n_classes = 150 

    for e in range(args.start, args.start + args.epochs):
        #if e >= 25:
        #    optim.param_groups[-1]['lr'] = args.lr * 0.1
        start = time.time()

        train_loss, train_acc  = trainer.train(train_load, e)
        test_loss, test_acc  = trainer.eval(test_load, e)
        np.savez("class_acc/base/cmatrix_{}".format(str(e)),test = test_class_acc, train=train_class_acc)

        t = time.time() - start
        lr = optim.param_groups[-1]['lr']
        logger.log(e, t, train_loss, train_acc, test_loss, test_acc, lr)
        checkpointer.checkpoint(net, test_loss, e)


if __name__ == '__main__':
    main()


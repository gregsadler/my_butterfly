import torch
import torchvision


class BFResnet(torch.nn.Module):
    '''Baseline Resnet for single-wing classification'''
    def __init__(self, nclasses=150):
        super(BFResnet, self).__init__()

        self.resnet = torchvision.models.resnet18(pretrained=True)
        fc = torch.nn.Linear(self.resnet.fc.in_features, nclasses)
        self.fc = fc
        delattr(self.resnet, 'fc')

    def no_fc(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = x.view(x.size()[0], -1)
        return x

    def forward(self, x):
        x = self.no_fc(x)
        return self.fc(x)

# ************************************************************************** #
# ************************************************************************** #

class _SingleBranch(torch.nn.Module):
    def __init__(self, nclasses=150, d_emb_in=512, d_emb_out=512,
                 single_layer=False):
        super(_SingleBranch, self).__init__()

        self.single = single_layer

        self.embedding = torch.nn.Linear(d_emb_in, d_emb_out)
        self.bn = torch.nn.BatchNorm1d(d_emb_out)
        self.relu = torch.nn.LeakyReLU(inplace=True)
        self.fc = torch.nn.Linear(d_emb_out, nclasses)

    def forward(self, x):
        #if not self.single:
        x = self.embedding(x)
        x = self.relu(self.bn(x))
        x = x / x.norm(p=2, dim=1, keepdim=True)
        return x, self.fc(x)


class _BranchNet(torch.nn.Module):
    def __init__(self, branches=1, nclasses=150, d_emb_in=512, d_emb_out=512,
                 single_layer=False):
        super(_BranchNet, self).__init__()

        self.branches = torch.nn.ModuleList([
            _SingleBranch(nclasses, d_emb_in, d_emb_out, single_layer)
            for i in range(branches)])

    def forward(self, x):
        xs = [branch(x) for branch in self.branches]

        embs = torch.stack([x[0] for x in xs], 1).squeeze_()
        preds = torch.stack([x[1] for x in xs], 1).squeeze_()

        return embs, preds


class ResnetBase(torch.nn.Module):
    '''Resnet without the final fully connected layer.
    
    args:
        model (str): The resnet model to use, e.g. resnet18 or resnet50.
    '''
    def __init__(self, model='resnet18'):
        super(ResnetBase, self).__init__()

        model = getattr(torchvision.models, model)
        self.resnet = model(pretrained=True)
        self.final_dim = self.resnet.fc.in_features
        delattr(self.resnet, 'fc')

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)

        return x
        

class Wingnet(torch.nn.Module):
    '''Resnet model with branches for multiple wings.'''
    def __init__(self, nclasses=150, branches=1, invar=True, out_emb=True,
                 model='resnet18'):
        super(Wingnet, self).__init__()

        self.out_emb = out_emb

        self.resnet = ResnetBase(model)

        self.d_emb = self.resnet.final_dim

        self.head = _BranchNet(branches, nclasses, self.d_emb, self.d_emb,single_layer= not invar)

    def freeze_base_(self):
        for p in self.resnet.parameters():
            p.requires_grad = False
        self.resnet.eval()

    def forward(self, x):
        x = self.resnet(x)
        emb, x = self.head(x)
        
        if self.out_emb:
            # returns embedding and predictions
            # both are 3D tensors [B x 4 x *]
            return emb, x
        else:
            return x


class WingnetA(torch.nn.Module):
    '''Multiple-wing aggregation on wingnet'''
    def __init__(self, nparts=(8,1), nclasses=150, invar=True, model='resnet18'):
        super(WingnetA, self).__init__()

        self.basenet = Wingnet(nclasses, 4, invar, model)

        self.aggregate_branch = torch.nn.Sequential(
            torch.nn.Linear(2 * (nparts[0] + nparts[1]), 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, nparts[0]),
            torch.nn.Sigmoid()
        )

    def forward(self, stats, part_img, valid_part_mask):
        
        part_pred = self.basenet.forward(part_img)[1]

        weight = self.aggregate_branch(stats)
        weight = weight * valid_part_mask
        weight = weight / weight.norm(1, 1, keepdim=True)
        
        #pred = 

        return weight, part_pred


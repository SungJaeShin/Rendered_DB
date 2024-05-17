import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import h5py
import pdb

class NetVLADLayer(nn.Module):
    """NetVLAD Module Implementation"""

    def __init__(self, config):
        """
        Args
            args.num_clusters : Number of clusters
            args.dim = channel-wise dimension of feature space
        """
        super(NetVLADLayer, self).__init__()
        self.num_clusters = config['model']['num_clusters']
        self.encoder_dim = config['model']['encoder_dim']
        # self.centroid_cache = config['cacheroot']['centroid_cache']
        self.alpha = 0
        self.centroids = nn.Parameter(torch.rand(self.num_clusters, self.encoder_dim))
        self.soft_assign = nn.Conv2d(self.encoder_dim, self.num_clusters, kernel_size=1, bias=False)


    def init_parameters(self):
        # Initialize: centroids
        print("=====> Initializing centroids and alpha using: ", self.centroid_cache)
        with h5py.File(self.centroid_cache, mode='r') as h5:
            centroids = h5.get("centroids")[...]
            descriptors = h5.get("descriptors")[...]

        self.centroids = nn.Parameter(torch.from_numpy(centroids))


        # Initialize: alpha (Refer Appendix.A in the paper)
        dots = np.dot(centroids, descriptors.T) #  equivalent to 2*alpla*(c_k * x_i) part
        dots.sort(0)
        dots = dots[::-1, :] # Sort in descending order
        self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()

        # Initialize: soft-assigning pointwise convolution
        self.soft_assign.weight = nn.Parameter(torch.from_numpy(self.alpha*centroids).unsqueeze(2).unsqueeze(3))

    def forward(self, x):
        B, C = x.shape[:2]

        x = F.normalize(x, p=2, dim=1)
        soft_assign = self.soft_assign(x).view(B, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(B, C, -1)

        vlad = torch.zeros([B, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)

        for c in range(self.num_clusters):
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                self.centroids[c:c+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)

            residual *= soft_assign[:, c:c+1, :].unsqueeze(2)
            vlad[:, c:c+1, :] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)


class NetVLAD(nn.Module):
    def __init__(self, config):
        super(NetVLAD, self).__init__()
        self.encoder = self.buildEncoder(config)
        self.netvlad = NetVLADLayer(config)

    def buildEncoder(self, config):
        layers=[]
        if config['model']['backbone'] == 'resnet50':  # Last ReLU is included unlike configurations for VGG16
            backbone = models.resnet50(pretrained=True)
            layers = list(backbone.children())[:-2]
        elif config['model']['backbone'] == 'vgg16':
            backbone = models.vgg16(pretrained=True)
            layers = list(backbone.features.children())[:-2]
            for l in layers[:-5]: 
                for p in l.parameters():
                    p.requires_grad = False

        layers.append(L2Norm())
        encoder = nn.Sequential(*layers)

        return encoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.netvlad(x)
        return x
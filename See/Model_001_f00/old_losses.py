# def siim_loss(y_true, y_pred, weights=None, eps=1e-3):
#   assert y_true.shape == y_pred.shape
#   y_true_sum = y_true.sum((1, 2, 3))
#   non_empty = th.gt(y_true_sum, 0)
#   total_loss = 0.0
#   # dice loss
#   if non_empty.sum() > 0:
#     yt_non_empty, yp_non_empty = y_true[non_empty], y_pred[non_empty]
#     intersection = (yt_non_empty * yp_non_empty).sum((1, 2, 3))
#     dice = (2. * intersection) / (yt_non_empty.sum((1, 2, 3)) +
#         yp_non_empty.sum((1, 2, 3)))
#     dl = th.mean(1. - dice)
#     total_loss += 0.1 * dl

#   # bce loss
#   y_pred = th.clamp(y_pred, 1e-6, 1. - 1e-6)
#   bce = -y_true * th.log(y_pred) - (1. - y_true) * th.log(
#       1. - y_pred)
#   bce = th.mean(bce)
#   total_loss += bce

#   return total_loss


# def siim_loss(y_true, y_pred, weights=None, alpha=0.25, gamma=2.0):
#   assert y_true.shape == y_pred.shape
#   y_pred = th.clamp(y_pred, 1e-6, 1. - 1e-6)
#   loss = -y_true * alpha * (1. - y_pred) ** gamma * th.log(y_pred) - \
#       (1. - y_true) * (1. - alpha) * y_pred ** gamma * th.log(1. - y_pred)
#   loss = th.mean(loss)
#   return loss


# class EMAModule(nn.Module):
#     def __init__(self, channels, K, lbda=1, alpha=0.5, T=3):
#         super().__init__()
#         self.T = T
#         self.alpha = alpha
#         self.lbda = lbda
#         self.conv1_in = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
#         self.conv1_out = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
#         self.bn_out = nn.BatchNorm2d(channels)
#         self.bases = nn.Parameter(torch.empty(K, channels),
#             requires_grad=False)  # K x C
#         nn.init.kaiming_uniform_(self.bases, a=math.sqrt(5))
#         self.bases.data = F.normalize(self.bases.data, dim=-1)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         residual = x
#         x = self.conv1_in(x).view(B, C, -1).transpose(1, -1)  # B x N x C

#         bases = self.bases[None, ...]
#         x_in = x.detach()
#         for i in range(self.T):
#             # Expectation
#             if i == (self.T - 1):
#                 x_in = x
#             z = torch.softmax(self.lbda * torch.matmul(x_in,
#                 bases.transpose(1, -1)), dim=-1)  # B x N x K
#             # Maximization
#             bases = torch.matmul(z.transpose(1, 2), x_in) / (
#                 z.sum(1)[..., None] + 1e-12)  # B x K x C
#             bases = F.normalize(bases, dim=-1)

#         if self.training:
#             self.bases.data = self.alpha * self.bases + \
#                 (1 - self.alpha) * bases.detach().mean(0)

#         x = torch.matmul(z, bases).transpose(1, -1).view(B, C, H, W)
#         x = self.conv1_out(x)
#         x = self.bn_out(x)
#         x += residual
#         return x


# class EMAModule(nn.Module):
#     def __init__(self, channels, K, lbda=1, alpha=0.1, T=3):
#         super().__init__()
#         self.T = T
#         self.alpha = alpha
#         self.lbda = lbda
#         self.conv1_in = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
#         self.conv1_out = nn.Conv2d(channels, channels, kernel_size=1, padding=0,
#             bias=False)
#         self.bn_out = nn.BatchNorm2d(channels)
#         self.register_buffer('bases', torch.empty(K, channels))  # K x C
#         nn.init.kaiming_uniform_(self.bases)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         residual = x
#         x = self.conv1_in(x).view(B, C, -1).transpose(1, -1)  # B x N x C

#         bases = self.bases.unsqueeze(0)
#         x_in = x.detach()
#         for i in range(self.T):
#             # Expectation
#             if i == (self.T - 1):
#                 x_in = x
#             z = torch.softmax(self.lbda * torch.matmul(x_in,
#                 bases.transpose(1, -1)), dim=-1)  # B x N x K
#             # Maximization
#             bases = torch.matmul(z.transpose(1, 2), x_in) / (
#                 z.sum(1).unsqueeze(-1) + 1e-12)  # B x K x C
#             bases = F.normalize(bases, dim=-1)
#         if self.training:
#             self.bases.data = (1 - self.alpha) * self.bases + \
#                 self.alpha * bases.detach().mean(0)

#         x = torch.matmul(z, bases).transpose(1, -1).view(B, C, H, W)
#         x = self.conv1_out(x)
#         x = self.bn_out(x)
#         x += residual
#         return x


# class CARAFE(nn.Module):
#     def __init__(self, num_channels, scale_factor=2, Cm=64, k_encoder=3,
#                  k_up=5):
#         # https://arxiv.org/pdf/1905.02188.pdf
#         super().__init__()
#         self.num_channels = num_channels
#         self.scale_factor = scale_factor
#         self.Cm = Cm
#         self.k_encoder = k_encoder
#         self.k_up = k_up
#         self.C_up = self.scale_factor ** 2 * self.k_up ** 2
#         self.compressor = nn.Conv2d(num_channels, self.Cm, 1, bias=True)
#         self.encoder = nn.Conv2d(self.Cm, self.C_up, k_encoder,
#             padding=k_encoder // 2, bias=True)
#         self.pixel_shuffle = nn.PixelShuffle(self.scale_factor)

#     def forward(self, x):
#         N, C, H, W = x.shape
#         # kernel prediction module
#         compressed = self.compressor(x)  # N, Cm, H, W
#         kernels = self.encoder(compressed)  # N, k k s s, H, W
#         kernels = kernels.view(N, self.k_up ** 2, self.scale_factor ** 2, -1)  # N, k k, s s, H W  # noqa
#         kernels = F.softmax(kernels, 1)

#         # content-aware reassembly module
#         x = F.unfold(x, (self.k_up, self.k_up), padding=self.k_up // 2)  # N, C k k, H W  # noqa
#         x = x.view(N, C, self.k_up ** 2, -1)  # N, C, k k, H W
#         kernels = kernels.to(x.dtype)
#         x = torch.einsum('nckh,nksh->ncsh', x, kernels)
#         x = F.pixel_shuffle(x.view(N, -1, H, W), self.scale_factor)
#         return x


# class SelfAttention(nn.Module):
#   '''
#   The basic implementation for self-attention block/non-local block
#   https://github.com/PkuRainBow/OCNet.pytorch/blob/master/oc_module/base_oc_block.py
#   Input:
#       N X C X H X W
#   Parameters:
#       in_channels       : the dimension of the input feature map
#       key_channels      : the dimension after the key/query transform
#       value_channels    : the dimension after the value transform
#       scale             : choose the scale to downsample the input feature
#                           maps (save memory cost)
#   Return:
#       N X C X H X W
#       position-aware context features.(w/o concate or add with the input)
#   '''
#   def __init__(self, in_channels, key_channels, value_channels,
#                out_channels=None, scale=1):
#     super().__init__()
#     self.scale = scale
#     self.in_channels = in_channels
#     self.out_channels = out_channels
#     self.key_channels = key_channels
#     self.value_channels = value_channels
#     if out_channels is None:
#         self.out_channels = in_channels
#     self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
#     self.phi = nn.Sequential(
#         nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
#             kernel_size=1, stride=1, padding=0, bias=False),
#         nn.BatchNorm2d(self.key_channels),
#         nn.ReLU(inplace=True)
#     )
#     self.theta = nn.Sequential(
#         nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
#             kernel_size=1, stride=1, padding=0, bias=False),
#         nn.BatchNorm2d(self.key_channels),
#         nn.ReLU(inplace=True)
#     )
#     self.g = nn.Sequential(
#         nn.Conv2d(in_channels=self.in_channels,
#             out_channels=self.value_channels, kernel_size=1, stride=1,
#             padding=0, bias=False),
#         nn.BatchNorm2d(self.value_channels),
#         nn.ReLU(inplace=True)
#     )

#   def forward(self, x):
#     if self.scale > 1:
#       x = self.pool(x)

#     N, C, H, W = x.shape
#     query = self.theta(x).view(N, self.key_channels, -1)
#     query = query.permute(0, 2, 1)
#     key = self.phi(x).view(N, self.key_channels, -1)

#     A = torch.matmul(query, key)
#     A = (self.key_channels ** -.5) * A
#     A = F.softmax(A, dim=-1)

#     value = self.g(x).view(N, self.value_channels, -1)
#     value = value.permute(0, 2, 1)
#     Z = torch.matmul(A, value)
#     Z = Z.permute(0, 2, 1).contiguous()
#     Z = Z.view(N, self.value_channels, *x.size()[2:])
#     if self.scale > 1:
#       Z = torch.nn.functional.interpolate(Z,
#           scale_factor=self.scale, mode='bilinear', align_corners=False)
#     return Z


# class ISA(nn.Module):
#   def __init__(self, in_channels, P_h=8, P_w=8, residual=True):
#     super().__init__()
#     self.in_channels = in_channels
#     self.P_h = P_h
#     self.P_w = P_w
#     self.residual = residual
#     self.lr_attention = SelfAttention(in_channels,
#         key_channels=in_channels // 2, value_channels=in_channels, scale=1)
#     self.sr_attention = SelfAttention(in_channels,
#         key_channels=in_channels // 2, value_channels=in_channels, scale=1)

#   def forward(self, x):
#     residual = x
#     N, C, H, W = x.shape
#     Q_h, Q_w = H // self.P_h, W // self.P_w
#     x = x.reshape(N, C, Q_h, self.P_h, Q_w, self.P_w)

#     # long-range attention
#     x = x.permute(0, 3, 5, 1, 2, 4)
#     x = x.reshape(N * self.P_h * self.P_w, C, Q_h, Q_w)
#     x = self.lr_attention(x)
#     x = x.reshape(N, self.P_h, self.P_w, C, Q_h, Q_w)

#     # short-range attention
#     x = x.permute(0, 4, 5, 3, 1, 2)
#     x = x.reshape(N * Q_h * Q_w, C, self.P_h, self.P_w)
#     x = self.sr_attention(x)
#     x = x.reshape(N, Q_h, Q_w, C, self.P_h, self.P_w)

#     x = x.permute(0, 3, 1, 4, 2, 5).reshape(N, C, H, W)
#     if self.residual:
#       x = x + residual
#     return x

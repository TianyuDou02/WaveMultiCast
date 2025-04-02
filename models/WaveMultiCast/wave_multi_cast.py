import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
import torch.nn.functional as F
from omegaconf import OmegaConf
from .mamba import VSSBlock

from .midlayer import ChannelAttention
from pytorch_wavelets import DWTForward
class BasicConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 upsampling=False,
                 act_norm=True,
                 act_inplace=False):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if upsampling is True:
            self.conv = nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels*4, kernel_size=kernel_size,
                          stride=1, padding=padding, dilation=dilation),
                nn.PixelShuffle(2)
            ])
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation)
        if self.act_norm:
            self.norm = nn.GroupNorm(2, out_channels)
            self.act = nn.SiLU(inplace=act_inplace)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            # trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y
class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        # self.conv_bn_relu = nn.Sequential(
        #                             nn.Conv2d(in_ch*4, out_ch, kernel_size=1, stride=1),
        #                             nn.BatchNorm2d(out_ch),   
        #                             nn.ReLU(inplace=True),                                 
        #                             ) 
        self.conv = nn.Conv2d(in_ch*4, out_ch, kernel_size=1, stride=1)
    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)        
        x = self.conv(x)

        return x
class WTSC(nn.Module):
    def __init__(self,c_in,c_out,act_norm=True,act_inplace=False,
                 upsampling=False,downsampling=False):
        super(WTSC,self).__init__()
        self.act_norm = act_norm
        if downsampling:
            self.sample = Down_wt(c_in,c_out)
        else:
            self.sample = nn.Conv2d(c_in,c_out,3,1,1)
            # self.sample = nn.Identity()
        if self.act_norm:
            self.norm = nn.GroupNorm(2,c_out)
            self.act = nn.SiLU(inplace=act_inplace)
    def forward(self,x):
        y = self.sample(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y

class ConvSC(nn.Module):

    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size=3,
                 downsampling=False,
                 upsampling=False,
                 act_norm=True,
                 act_inplace=False):
        super(ConvSC, self).__init__()

        stride = 2 if downsampling is True else 1
        padding = (kernel_size - stride + 1) // 2

        self.conv = BasicConv2d(C_in, C_out, kernel_size=kernel_size, stride=stride,
                                upsampling=upsampling, padding=padding,
                                act_norm=act_norm, act_inplace=act_inplace)

    def forward(self, x):
        y = self.conv(x)
        return y

class MambaDU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 downsampling=False,
                 upsampling=False,
                 act_norm=True,
                 act_inplace=False,
                 vss=True
                 ):
        super(MambaDU, self).__init__()
        stride = 2 if downsampling else 1
        padding = (kernel_size - stride + 1) // 2
        self.vss = vss
        self.conv = BasicConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, 
                                upsampling=upsampling, act_norm=act_norm, act_inplace=act_inplace,)
        if vss:
            self.VSS = VSSBlock(hidden_dim=out_channels)
    def forward(self, x):
        x = self.conv(x)
        if self.vss:
            y = x.permute(0,2,3,1).contiguous()
            y = self.VSS(y)
            y = y.permute(0,3,1,2).contiguous()
            return y+x
        return x

def sampling_generator(N, reverse=False):
    samplings = [False,True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]
def picture_generator(N,sampling,picture_size):
    pictures = []
    for i in range(N):
        if sampling[i]:
            pictures.append(picture_size//2)
            picture_size = picture_size//2
        else:
            pictures.append(picture_size)
    return pictures
class DWEncoder(nn.Module):
    def __init__(self,C_in,C_hid,N_S,act_inplace=False):
        super(DWEncoder,self).__init__()
        sampling = sampling_generator(N_S)
        self.enc = nn.Sequential(
            WTSC(C_in,C_hid,downsampling=sampling[0],act_inplace=act_inplace),
            *[WTSC(C_in if i==0 else C_hid,C_hid,downsampling=sampling[i],act_inplace=act_inplace) for i in range(1,N_S)]
        )
    def forward(self,x):
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1,len(self.enc)):
            latent = self.enc[i](latent)
        return latent,enc1

class Encoder(nn.Module):
    """3D Encoder for SimVP"""

    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
              ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0],
                     act_inplace=act_inplace,),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s,
                     act_inplace=act_inplace) for s in samplings[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1
class Decoder(nn.Module):
    """3D Decoder for SimVP"""

    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s,
                     act_inplace=act_inplace) for s in samplings[:-1]],
              ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1],
                     act_inplace=act_inplace,)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)
        return Y

class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', nn.BatchNorm2d(out_planes))
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)
class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.SiLU(inplace=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + x
        return x
class Star(nn.Module):
    def __init__(self,in_channels,out_channels,):
        super().__init__()
        self.convbn = ConvBN(in_channels,out_channels,3,1,1)
        self.block = Block(dim=out_channels)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward(self,x):
        x = self.convbn(x)
        x = self.block(x)
        return x

class MidSpatio(nn.Module):
    def __init__(self,in_channels,hid_channels,out_channels,depth):
        super(MidSpatio,self).__init__()
        self.depth = depth
        self.layers = []
        self.layers.append(Star(in_channels,hid_channels))
        for i in range(1,depth-1):
            self.layers.append(Star(hid_channels,hid_channels))
        self.layers.append(Star(hid_channels,out_channels))
        self.layers = nn.ModuleList(self.layers)
    def forward(self,x):
        x = self.layers[0](x)
        for i in range(1,self.depth-1):
            x = self.layers[i](x)+x
        return self.layers[-1](x)

class MambaCaMix(nn.Module):
    def __init__(self,in_channels,out_channels,):
        super(MambaCaMix,self).__init__()
        self.convin = nn.Conv2d(in_channels,out_channels,1)
        self.ca = ChannelAttention(out_channels)
        self.mamba = VSSBlock(out_channels)
        # self.convout = nn.Conv2d(out_channels*2,out_channels,1)
        self.norm = nn.GroupNorm(2,out_channels)
        self.act = nn.SiLU(False)
    def forward(self,x):
        x = self.convin(x)
        # ca = self.ca(x)
        m = x.permute(0,2,3,1)
        mamba = self.mamba(m)
        mamba = mamba.permute(0,3,1,2).contiguous()
        ca = self.ca(mamba)
        return self.act(self.norm(ca))+x
        # out = torch.cat((ca,mamba),dim=1)
        # out = ca+mamba
        # out = self.act(self.norm(self.convout(out)))
        return out+x

class MidMambaUnet(nn.Module):
    def __init__(self,in_channels,hid_channels,out_channels,depth):
        super(MidMambaUnet,self).__init__()
        self.depth = depth
        self.enc = []
        self.enc.append(MambaCaMix(in_channels=in_channels,out_channels=hid_channels))
        for i in range(1,depth):
            self.enc.append(MambaCaMix(in_channels=hid_channels,out_channels=hid_channels))

        self.dec = [MambaCaMix(in_channels=hid_channels,out_channels=hid_channels)]
        for i in range(1,depth-1):
            self.dec.append(MambaCaMix(in_channels=hid_channels*2,out_channels=hid_channels))
        self.dec.append(MambaCaMix(in_channels=hid_channels*2,out_channels=out_channels))
        self.enc = nn.ModuleList(self.enc)
        self.dec = nn.ModuleList(self.dec)
    def forward(self,x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.depth):
            z = self.enc[i](z)
            if i < self.depth-1:
                skips.append(z)
        # decoder
        z = self.dec[0](z)
        for i in range(1,self.depth):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1) )

        y = z.reshape(B, T, C, H, W)
        return y

    
class WaveMultiCast_Model(nn.Module):
    def __init__(self,in_shape, T_in, T_out,  **kwargs):
        super().__init__()
        C,H,W = in_shape
        self.T_in = T_in
        self.T_out = T_out
        self.alpha = configs.alpha
        self.beta = configs.beta

        hid_S , hid_T = configs.hid_S, configs.hid_T
        S_Ratio = configs.S_Ratio
        en_de_depth, mid_depth = configs.N_S, configs.N_T

        # self.enc = Encoder(C_in=C,C_hid=hid_S,N_S=en_de_depth,spatio_kernel=3,act_inplace=False)
        self.enc = DWEncoder(C_in=C,C_hid=hid_S,N_S=en_de_depth,act_inplace=False)
        self.dec = Decoder(C_hid=hid_S,C_out=C,N_S=en_de_depth,spatio_kernel=3,act_inplace=False)
        self.mid = MidMambaUnet(in_channels=T_in*hid_S,hid_channels=hid_T,out_channels=T_in*hid_S,depth=mid_depth)
        self.mid_spatio = MidSpatio(in_channels=hid_S,hid_channels=hid_T,out_channels=hid_S,depth=mid_depth)

        self.MSE_criterion = nn.MSELoss()
    def forward(self, x_raw, **kwargs):
        B, T, C, H, W = x_raw.shape
        x = x_raw.reshape(B*T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.mid(z)
        hid = hid.reshape(B*T, C_, H_, W_)
        hid_s = self.mid_spatio(embed)
        hid = hid+hid_s
        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)

        return Y
    # loss for diversity and regularization
    def diff_div_reg(self, pred_y, batch_y, tau=0.1, eps=1e-12):
        B, T, C = pred_y.shape[:3]
        if T <= 2:  return 0
        gap_pred_y = (pred_y[:, 1:] - pred_y[:, :-1]).reshape(B, T-1, -1)
        gap_batch_y = (batch_y[:, 1:] - batch_y[:, :-1]).reshape(B, T-1, -1)
        softmax_gap_p = F.softmax(gap_pred_y / tau, -1)
        softmax_gap_b = F.softmax(gap_batch_y / tau, -1)
        loss_gap = softmax_gap_p * \
            torch.log(softmax_gap_p / (softmax_gap_b + eps) + eps)
        return loss_gap.mean()
    # loss for exteme values
    def Exloss(self,pred, target, up_th=0.9, down_th=0.1, lamda_underestimate=1.2, lamda_overestimate=1.0, lamda=1.0):
        '''
        up_th: percentile threshold of maximum value
        down_th: percentile threshold of minimum value
        lamda_underestimate: The penalty when underestimating is greater than the penalty when overestimating
        lamda_overestimate: Penalty for overestimation
        lamda: weight of Exloss and MSE
        '''
        N, T, C, H, W = pred.shape
        # Get the 90% and 10% quantiles in target as the thresholds for extreme maximum and minimum values, denoted as tar_up and tar_down
        tar_up =  torch.quantile(target.view(N, T, C, H*W), q=up_th, dim=-1).unsqueeze(-1).unsqueeze(-1) # N,C,1,1
        tar_down =  torch.quantile(target.view(N, T, C, H*W), q=down_th, dim=-1).unsqueeze(-1).unsqueeze(-1) # N,C,1,1

        target_up_area = F.relu(target-tar_up) # The part of target that is greater than tar_up
        target_down_area = -F.relu(tar_down-target) # The part of target that is smaller than tar_down
        pred_up_area = F.relu(pred-tar_up) # The part of pred that is greater than tar_up
        pred_down_area = -F.relu(tar_down-pred) # The part of pred that is smaller than tar_down

        # Increase the loss weight for the underestimated part of pred (the maximum value prediction is too small, the minimum value prediction is too large)
        loss_up = lamda_underestimate*(target_up_area-pred_up_area)*F.relu(target_up_area-pred_up_area)+\
                lamda_overestimate*(pred_up_area-target_up_area)*F.relu(pred_up_area-target_up_area)
        loss_down = lamda_overestimate*(target_down_area-pred_down_area)*F.relu(target_down_area-pred_down_area)+\
                    lamda_underestimate*(pred_down_area-target_down_area)*F.relu(pred_down_area-target_down_area)
        loss_up = torch.mean(loss_up)
        loss_down = torch.mean(loss_down)
        ex_loss = (loss_up + loss_down)/(1-up_th+down_th)
        return lamda*ex_loss
    def predict(self, frames_in, frames_gt=None, compute_loss=False, **kwargs):
        frames_pred = []
        cur_seq = frames_in.clone()
        for _ in range(self.T_out // self.T_in):
            cur_seq = self.forward(cur_seq)
            frames_pred.append(cur_seq)
        
        frames_pred = torch.cat(frames_pred, dim=1)
        if compute_loss:
            loss = self.MSE_criterion(frames_pred, frames_gt)+self.alpha*self.diff_div_reg(frames_pred, frames_gt)+self.Exloss(frames_pred,frames_gt,lamda=self.beta)
        else:
            loss = None
        return frames_pred, loss


config_dict = {
    "hid_S" : 64,
    "hid_T" : 256,
    "S_Ratio" : 1,
    "N_T": 4,
    "N_S": 2,
    "alpha": 0.1,
    "beta": 1.0,
}

configs = OmegaConf.create(config_dict)

    

def get_model(in_shape, T_in, T_out, **kwargs):
    return WaveMultiCast_Model(in_shape, T_in=T_in, T_out=T_out, **kwargs)

if __name__ == "__main__":
    
    in_shape = (1, 128, 128)
    model = WaveMultiCast_Model(in_shape, T_in=5, T_out=20).to('cuda')
    frames_in = torch.randn(1, 5, 1, 128, 128).to('cuda')
    frames_gt = torch.randn(1, 20, 1, 128, 128).to('cuda')
    # frames_gt = torch.randn(2, 10, 1, 128, 128)
    # mask_true = torch.zeros(2, 15, 1, 128, 128)
    # print(torch.any(torch.isnan(x)))
    frames_pred, loss = model.predict(frames_in=frames_in, frames_gt=frames_gt, compute_loss=True)
    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is None :
            print(name)
    print(frames_pred.shape)
    print(loss)
    total_params = sum(p.numel() for p in model.parameters())  
    total_size = total_params * 4 / (1024 ** 2)  
    print(f"Model size: {total_size:.2f} MB")
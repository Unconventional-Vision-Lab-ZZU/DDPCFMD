"""Transformer, sampling, fusion, and estimation layers for DDPCFMD."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint
from .normalization import spatial_normalization


def spatial_pair(value):
    """Convert a scalar spatial size to a height-width pair."""
    return value if isinstance(value, (tuple, list)) else (value, value)


class EH(nn.Sequential):
    """Estimation head: 1x1 projection followed by sigmoid."""

    def __init__(self, in_channels, out_channels):
        super().__init__(nn.Conv2d(in_channels, out_channels, 1), nn.Sigmoid())


class DFFM(nn.Sequential):
    """Deep Feature Fusion Mechanism with local convolutions."""

    def __init__(self, in_channels, out_channels, norm_type="instance", bias=True):
        norm = spatial_normalization(norm_type)
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 1, bias=bias),
            norm(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=bias),
            norm(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=bias),
            norm(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )


class PatchEmbedding(nn.Module):

    def __init__(self, image_size=224, patch_size=4, in_chans=3, token_dim=96, norm_layer=None):
        super().__init__()
        image_size = spatial_pair(image_size)
        patch_size = spatial_pair(patch_size)
        patch_resolution = [image_size[0] // patch_size[0], image_size[1] // patch_size[1]]
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_resolution = patch_resolution
        self.num_patches = patch_resolution[0] * patch_resolution[1]
        self.in_chans = in_chans
        self.token_dim = token_dim
        self.proj = nn.Conv2d(in_chans, token_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(token_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, channels, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class RescaleLayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-05, detach_grad=False):
        super(RescaleLayerNorm, self).__init__()
        self.eps = eps
        self.detach_grad = detach_grad
        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))
        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)
        nn.init.trunc_normal_(self.meta1.weight, std=0.02)
        nn.init.constant_(self.meta1.bias, 1)
        nn.init.trunc_normal_(self.meta2.weight, std=0.02)
        nn.init.constant_(self.meta2.bias, 0)

    def forward(self, input):
        mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)
        normalized_input = (input - mean) / std
        if self.detach_grad:
            rescale, rebias = (self.meta1(std.detach()), self.meta2(mean.detach()))
        else:
            rescale, rebias = (self.meta1(std), self.meta2(mean))
        out = normalized_input * self.weight + self.bias
        return (out, rescale, rebias)


class SPM(nn.Module):
    """Slice Patch Merging."""

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        """Merge tokens shaped (B, H * W, C) from four spatial offsets."""
        H, W = self.input_resolution
        B, L, channels = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = x.view(B, H, W, channels)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * channels)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class DGU(nn.Module):
    """Dual-sampling Gating Upsampling with pixel-shuffle and bilinear branches."""

    def __init__(self, input_resolution, in_channels, scale_factor):
        super(DGU, self).__init__()
        self.input_resolution = input_resolution
        self.factor = scale_factor
        if self.factor == 2:
            self.gate = nn.Conv2d(in_channels, 2, 3, 1, 1, bias=True)
            self.conv = nn.Conv2d(in_channels, in_channels // 2, 1, 1, 0, bias=False)
            self.up_p = nn.Sequential(
                nn.Conv2d(in_channels, 2 * in_channels, 1, 1, 0, bias=False),
                nn.PReLU(),
                nn.PixelShuffle(scale_factor),
                nn.Conv2d(in_channels // 2, in_channels // 2, 1, stride=1, padding=0, bias=False),
            )
            self.up_b = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1, 1, 0),
                nn.PReLU(),
                nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0, bias=False),
            )
        # Output mode doubles spatial size and retains the channel width.
        elif self.factor == 4:
            self.gate = nn.Conv2d(in_channels, 2, 3, 1, 1, bias=True)
            self.conv = nn.Conv2d(in_channels // 2, in_channels, 1, 1, 0, bias=False)
            self.up_p = nn.Sequential(
                nn.Conv2d(in_channels, 2 * in_channels, 1, 1, 0, bias=False),
                nn.PReLU(),
                nn.PixelShuffle(2),
                nn.Conv2d(in_channels // 2, in_channels // 2, 1, stride=1, padding=0, bias=False),
            )
            self.up_b = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1, 1, 0),
                nn.PReLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0, bias=False),
            )

    def forward(self, x):
        """Upsample tokens shaped (B, H * W, C)."""
        if type(self.input_resolution) == int:
            H = self.input_resolution
            W = self.input_resolution
        elif type(self.input_resolution) == tuple:
            H, W = self.input_resolution
        B, L, channels = x.shape
        x = x.view(B, H, W, channels)
        x = x.permute(0, 3, 1, 2)
        x_p = self.up_p(x)
        x_b = self.up_b(x)
        gates = self.gate(torch.cat((x_p, x_b), dim=1))
        gated_y = x_p * gates[:, [0], :, :] + x_b * gates[:, [1], :, :]
        if self.factor == 4:
            gated_y = self.conv(gated_y)
        out = gated_y.permute(0, 2, 3, 1)
        if self.factor == 2:
            out = out.view(B, -1, channels // 2)
        return out


class ESTBDecoderStage(nn.Module):

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        upsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                ESTB(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        if upsample is not None:
            self.upsample = DGU(input_resolution, in_channels=dim, scale_factor=2)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        multiscale_feature = x
        if self.upsample is not None:
            x = self.upsample(x)
        return (x, multiscale_feature)


class ESTBEncoderStage(nn.Module):

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                ESTB(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
            multiscale_feature = x
        if self.downsample is not None:
            x = self.downsample(x)
        return (x, multiscale_feature)


class FDFM(nn.Module):
    """Frequency-domain Decomposition and Fusion Module."""

    def __init__(self, dim, input_resolution, num_heads, window_size, shift_size):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.attn = WindowSelfAttention(dim, window_size, num_heads)
        self.register_buffer("attn_mask", None, persistent=False)
        self.conv = nn.Conv2d(
            dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode="reflect"
        )
        self.V = nn.Conv2d(dim, dim, 1)
        self.QK = nn.Conv2d(dim, dim * 2, 1)
        self.pointwise_conv1 = nn.Conv2d(dim, dim, 1)
        self.pointwise_conv2 = nn.Conv2d(dim, dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, channels, H, W = x.shape
        V = self.V(x)
        QK = self.QK(x)
        QKV = torch.cat([QK, V], dim=1)
        shifted_QKV = self.pad_attention_windows(QKV, self.shift_size > 0)
        Ht, Wt = shifted_QKV.shape[2:]
        if self.shift_size > 0 and (
            self.attn_mask is None
            or self.attn_mask.shape[-1] != self.window_size**2
            or self.attn_mask.device != x.device
        ):
            self.attn_mask = self.build_attention_mask(Ht, Wt, x.device)
        shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
        qkv = partition_windows(shifted_QKV, self.window_size)
        attn_windows = self.attn(qkv, self.attn_mask)
        shifted_out = merge_windows(attn_windows, self.window_size, Ht, Wt)
        out = shifted_out[
            :, self.shift_size : self.shift_size + H, self.shift_size : self.shift_size + W, :
        ]
        attn_out = out.permute(0, 3, 1, 2)
        conv_out = self.conv(V)
        addition = conv_out + attn_out
        weight = F.avg_pool2d(addition, addition.size()[2:]).view(
            addition.size(0), addition.size(1), 1, 1
        )
        weight = self.softmax(weight)
        x = self.pointwise_conv1(addition) * weight + self.pointwise_conv2(addition) * (1 - weight)
        return x

    def build_attention_mask(self, H, W, device):
        if self.shift_size == 0:
            return None
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = partition_windows(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows[0].unsqueeze(0) - mask_windows[0].unsqueeze(1)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )
        attn_mask = attn_mask.unsqueeze(0).repeat(self.num_heads, 1, 1)
        return attn_mask

    def pad_attention_windows(self, x, shift=False):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        if shift:
            x = F.pad(
                x,
                (
                    self.shift_size,
                    (self.window_size - self.shift_size + mod_pad_w) % self.window_size,
                    self.shift_size,
                    (self.window_size - self.shift_size + mod_pad_h) % self.window_size,
                ),
                mode="reflect",
            )
        else:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        return x


class ESTB(nn.Module):
    """Enhanced Swin Transformer Block."""

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CPRC(in_features=dim, hidden_features=mlp_hidden_dim)
        self.fdfm = FDFM(dim, input_resolution, num_heads, self.window_size, self.shift_size)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, channels = x.shape
        x = x.view(B, H, W, channels).permute(0, 3, 1, 2)
        shortcut = x
        x, rescale, rebias = self.norm1(x)
        x = self.fdfm(x)
        x = x * rescale + rebias
        x = shortcut + x
        shortcut = x
        x, rescale, rebias = self.norm2(x)
        x = self.mlp(x)
        x = x * rescale + rebias
        x = shortcut + x
        x = x.view(B, channels, H * W).permute(0, 2, 1)
        return x


class WindowSelfAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** (-0.5)
        relative_positions = self.relative_position_coordinates(window_size)
        self.register_buffer("relative_positions", relative_positions)
        self.meta = nn.Sequential(
            nn.Linear(2, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_heads, bias=True),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, attn_mask=None):
        """Attend to qkv shaped (window_batch, tokens, 3 * dim).

        attn_mask has shape (num_heads, tokens, tokens), or is None.
        """
        B_, N, _ = qkv.shape
        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.meta(self.relative_positions)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if attn_mask is not None:
            attn = attn + attn_mask.unsqueeze(0)
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
        return x

    def relative_position_coordinates(self, window_size):
        coords = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords, coords], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_positions = relative_positions.permute(1, 2, 0).contiguous()
        relative_positions_log = torch.sign(relative_positions) * torch.log1p(
            relative_positions.abs().float()
        )
        return relative_positions_log


def merge_windows(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def partition_windows(x, window_size):
    B, H, W, channels = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, channels)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size**2, channels)
    return windows


class CPRC(nn.Module):
    """Conv-PReLU-Conv feed-forward component."""

    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.PReLU(),
            nn.Conv2d(hidden_features, out_features, 1),
        )

    def forward(self, x):
        return self.mlp(x)


def spatial_projection(in_channels, out_channels, kernel_size, stride=1, bias=False):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=kernel_size // 2, stride=stride, bias=bias
    )

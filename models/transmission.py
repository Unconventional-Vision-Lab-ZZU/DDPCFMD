"""Polarization and TGI transmission estimation."""

import functools
import torch
from torch import nn
from .normalization import spatial_normalization
from .layers import (
    PatchEmbedding,
    RescaleLayerNorm,
    SPM,
    DGU,
    ESTBEncoderStage,
    ESTBDecoderStage,
    spatial_projection,
    EH,
    DFFM,
)


class TransmissionEstimation(nn.Module):
    """N1/N4 parameter estimation and transmission fusion."""

    def __init__(self, image_size=(240, 320)):
        super(TransmissionEstimation, self).__init__()
        feature_width = 32
        norm_type = "instance"
        channels = 3
        token_dim = 32
        patch_size = 2
        norm_layer2 = nn.LayerNorm
        depths = (2, 2, 2)
        num_heads = (2, 8, 16)
        window_size = 10
        mlp_ratio = (2, 2, 4)
        qkv_bias = True
        qk_scale = None
        patch_norm = True
        drop_rate = 0.0
        attn_drop_rate = 0.0
        drop_path_rate = 0.1
        bias = False
        self.patch_norm = patch_norm
        self.num_layers = len(depths)
        self.num_features = int(token_dim * 2 ** (self.num_layers - 1))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.token_dim = token_dim
        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_chans=token_dim,
            token_dim=token_dim,
            norm_layer=norm_layer2 if self.patch_norm else None,
        )
        patch_resolution = self.patch_embedding.patch_resolution
        self.patch_resolution = patch_resolution
        norm_layer = spatial_normalization(norm_type)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        self.airlight_dop_eh = EH(feature_width, channels)
        self.transmission_dop_eh = EH(feature_width, channels)
        self.tgi_transmission_eh = EH(feature_width, 3)
        self.dffm = DFFM(6, feature_width, norm_type=norm_type, bias=use_bias)
        self.transmission_eh = EH(feature_width, channels)
        self.n4_input_projection = nn.Sequential(
            spatial_projection(1, out_channels=32, kernel_size=3, bias=bias)
        )
        self.image_dgu = DGU(
            input_resolution=(patch_resolution[0], patch_resolution[1]),
            in_channels=token_dim,
            scale_factor=4,
        )
        self.n4_encoder = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ESTBEncoderStage(
                dim=int(token_dim * 2**i_layer),
                input_resolution=(
                    patch_resolution[0] // 2**i_layer,
                    patch_resolution[1] // 2**i_layer,
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio[i_layer],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=RescaleLayerNorm,
                downsample=SPM if i_layer < self.num_layers - 1 else None,
                use_checkpoint=False,
            )
            self.n4_encoder.append(layer)
        self.n4_decoder = nn.ModuleList()
        self.n4_skip_projection = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = (
                nn.Linear(
                    2 * int(token_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    int(token_dim * 2 ** (self.num_layers - 1 - i_layer)),
                )
                if i_layer > 0
                else nn.Identity()
            )
            layer_up = ESTBDecoderStage(
                dim=int(token_dim * 2 ** (self.num_layers - 1 - i_layer)),
                input_resolution=(
                    patch_resolution[0] // 2 ** (self.num_layers - 1 - i_layer),
                    patch_resolution[1] // 2 ** (self.num_layers - 1 - i_layer),
                ),
                depth=depths[self.num_layers - 1 - i_layer],
                num_heads=num_heads[self.num_layers - 1 - i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio[self.num_layers - 1 - i_layer],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[: self.num_layers - 1 - i_layer]) : sum(
                        depths[: self.num_layers - 1 - i_layer + 1]
                    )
                ],
                norm_layer=RescaleLayerNorm,
                upsample=DGU if i_layer < self.num_layers - 1 else None,
                use_checkpoint=False,
            )
            self.n4_decoder.append(layer_up)
            self.n4_skip_projection.append(concat_linear)
        self.n4_encoder_norm = norm_layer2(self.num_features)
        self.n4_decoder_norm = norm_layer2(self.token_dim)
        self.n1_encoder = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_rgb = ESTBEncoderStage(
                dim=int(token_dim * 2**i_layer),
                input_resolution=(
                    patch_resolution[0] // 2**i_layer,
                    patch_resolution[1] // 2**i_layer,
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio[i_layer],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=RescaleLayerNorm,
                downsample=SPM if i_layer < self.num_layers - 1 else None,
                use_checkpoint=False,
            )
            self.n1_encoder.append(layer_rgb)
        self.n1_decoder = nn.ModuleList()
        self.n1_skip_projection = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear_rgb = (
                nn.Linear(
                    2 * int(token_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    int(token_dim * 2 ** (self.num_layers - 1 - i_layer)),
                )
                if i_layer > 0
                else nn.Identity()
            )
            layer_up_rgb = ESTBDecoderStage(
                dim=int(token_dim * 2 ** (self.num_layers - 1 - i_layer)),
                input_resolution=(
                    patch_resolution[0] // 2 ** (self.num_layers - 1 - i_layer),
                    patch_resolution[1] // 2 ** (self.num_layers - 1 - i_layer),
                ),
                depth=depths[self.num_layers - 1 - i_layer],
                num_heads=num_heads[self.num_layers - 1 - i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio[self.num_layers - 1 - i_layer],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[: self.num_layers - 1 - i_layer]) : sum(
                        depths[: self.num_layers - 1 - i_layer + 1]
                    )
                ],
                norm_layer=RescaleLayerNorm,
                upsample=DGU if i_layer < self.num_layers - 1 else None,
                use_checkpoint=False,
            )
            self.n1_decoder.append(layer_up_rgb)
            self.n1_skip_projection.append(concat_linear_rgb)
        self.n1_encoder_norm = norm_layer(self.num_features)
        self.n1_decoder_norm = norm_layer(self.token_dim)
        self.n1_input_projection = nn.Sequential(
            spatial_projection(9, out_channels=32, kernel_size=3, bias=bias)
        )
        self.f1_encoder = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_fusion = ESTBEncoderStage(
                dim=int(token_dim * 2**i_layer),
                input_resolution=(
                    patch_resolution[0] // 2**i_layer,
                    patch_resolution[1] // 2**i_layer,
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio[i_layer],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=RescaleLayerNorm,
                downsample=SPM if i_layer < self.num_layers - 1 else None,
                use_checkpoint=False,
            )
            self.f1_encoder.append(layer_fusion)
        self.f1_decoder = nn.ModuleList()
        self.f1_skip_projection = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear_fusion = (
                nn.Linear(
                    2 * int(token_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    int(token_dim * 2 ** (self.num_layers - 1 - i_layer)),
                )
                if i_layer > 0
                else nn.Identity()
            )
            layer_up_fusion = ESTBDecoderStage(
                dim=int(token_dim * 2 ** (self.num_layers - 1 - i_layer)),
                input_resolution=(
                    patch_resolution[0] // 2 ** (self.num_layers - 1 - i_layer),
                    patch_resolution[1] // 2 ** (self.num_layers - 1 - i_layer),
                ),
                depth=depths[self.num_layers - 1 - i_layer],
                num_heads=num_heads[self.num_layers - 1 - i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio[self.num_layers - 1 - i_layer],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[: self.num_layers - 1 - i_layer]) : sum(
                        depths[: self.num_layers - 1 - i_layer + 1]
                    )
                ],
                norm_layer=RescaleLayerNorm,
                upsample=DGU if i_layer < self.num_layers - 1 else None,
                use_checkpoint=False,
            )
            self.f1_decoder.append(layer_up_fusion)
            self.f1_skip_projection.append(concat_linear_fusion)
        self.f1_encoder_norm = norm_layer(self.num_features)
        self.f1_decoder_norm = norm_layer(self.token_dim)

    def encode_tgi(self, x):
        encoder_feature = []
        residual = x
        x = self.patch_embedding(x)
        x_downsample = []
        for layer in self.n4_encoder:
            x_downsample.append(x)
            x, multiscale_feature = layer(x)
            encoder_feature.append(multiscale_feature)
        x = self.n4_encoder_norm(x)
        return (x, residual, x_downsample, encoder_feature)

    def decode_tgi(self, x, x_downsample):
        decoder_feature = []
        for inx, layer_up in enumerate(self.n4_decoder):
            if inx == 0:
                x, multiscale_feature = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[2 - inx]], -1)
                x = self.n4_skip_projection[inx](x)
                x, multiscale_feature = layer_up(x)
            decoder_feature.insert(0, multiscale_feature)
        x = self.n4_decoder_norm(x)
        return (x, decoder_feature)

    def decode_image_features(self, x):
        x = self.image_dgu(x)
        x = x.permute(0, 3, 1, 2)
        return x

    def encode_vlp(self, x):
        encoder_feature = []
        residual = x
        x = self.patch_embedding(x)
        x_downsample = []
        for layer in self.n1_encoder:
            x_downsample.append(x)
            x, multiscale_feature = layer(x)
            encoder_feature.append(multiscale_feature)
        x = self.n1_encoder_norm(x)
        return (x, residual, x_downsample, encoder_feature)

    def decode_vlp(self, x, x_downsample):
        decoder_feature = []
        for inx, layer_up in enumerate(self.n1_decoder):
            if inx == 0:
                x, multiscale_feature = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[2 - inx]], -1)
                x = self.n1_skip_projection[inx](x)
                x, multiscale_feature = layer_up(x)
            decoder_feature.insert(0, multiscale_feature)
        x = self.n1_decoder_norm(x)
        return (x, decoder_feature)

    def encode_transmission_fusion(self, x):
        encoder_feature = []
        residual = x
        x = self.patch_embedding(x)
        x_downsample = []
        for layer in self.f1_encoder:
            x_downsample.append(x)
            x, multiscale_feature = layer(x)
            encoder_feature.append(multiscale_feature)
        x = self.f1_encoder_norm(x)
        return (x, residual, x_downsample, encoder_feature)

    def decode_transmission_fusion(self, x, x_downsample):
        decoder_feature = []
        for inx, layer_up in enumerate(self.f1_decoder):
            if inx == 0:
                x, multiscale_feature = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[2 - inx]], -1)
                x = self.f1_skip_projection[inx](x)
                x, multiscale_feature = layer_up(x)
            decoder_feature.insert(0, multiscale_feature)
        x = self.f1_decoder_norm(x)
        return (x, decoder_feature)

    def forward(self, vlp, intensity, polarization_difference, tgi):
        gated_fea1 = self.n4_input_projection(tgi)
        x_gated, residual, x_downsample, gated_encode_feature = self.encode_tgi(gated_fea1)
        x_gated2, gated_decode_feature = self.decode_tgi(x_gated, x_downsample)
        T_hat1 = self.decode_image_features(x_gated2)
        tgi_transmission = self.tgi_transmission_eh(T_hat1)
        rgb_fea1 = self.n1_input_projection(vlp)
        x, residual, x_downsample, rgb_encode_feature = self.encode_vlp(rgb_fea1)
        backbone_out1, rgb_decode_feature = self.decode_vlp(x, x_downsample)
        backbone_out1 = self.decode_image_features(backbone_out1)
        airlight_dop = self.airlight_dop_eh(backbone_out1)
        transmission_dop = self.transmission_dop_eh(backbone_out1)
        # Polarization model: T = (I * P - I * P_A) / (P_T - P_A).
        physical_transmission = torch.clamp(
            (polarization_difference - intensity * airlight_dop)
            / (transmission_dop - airlight_dop + 1e-07),
            min=0,
            max=1,
        )
        cat2 = torch.cat([physical_transmission, tgi_transmission], dim=1)
        feature2 = self.dffm(cat2)
        x, residual, x_downsample, rgb_encode_feature = self.encode_transmission_fusion(feature2)
        backbone_out3, rgb_decode_feature = self.decode_transmission_fusion(x, x_downsample)
        backbone_out3 = self.decode_image_features(backbone_out3)
        transmission = self.transmission_eh(backbone_out3)
        return (airlight_dop, transmission_dop, tgi_transmission, transmission)

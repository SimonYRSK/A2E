import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import pandas as pd
import numpy as np
from datetime import datetime
from timm.layers.helpers import to_2tuple
from timm.models.swin_transformer_v2 import SwinTransformerV2Stage
from einops import rearrange

from models.dann import DomainClassifier

class ModuleFactory:
    def create_block(dim, out_dim, depth, input_resolution, window_size, **kwargs):


        return SwinTransformerV2Stage(
            dim= dim,
            out_dim = out_dim,
            window_size=window_size,
            depth=depth,
            input_resolution=input_resolution,  # 固定分辨率！
            num_heads=kwargs.get("num_heads", 8),
            use_checkpoint=kwargs.get("use_checkpoint", False)
        )

def get_pad3d(input_resolution, window_size):
    Pl, Lat, Lon = input_resolution
    win_pl, win_lat, win_lon = window_size

    padding_left = padding_right = padding_top = padding_bottom = padding_front = padding_back = 0
    pl_remainder = Pl % win_pl
    lat_remainder = Lat % win_lat
    lon_remainder = Lon % win_lon

    if pl_remainder:
        pl_pad = win_pl - pl_remainder
        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front
    if lat_remainder:
        lat_pad = win_lat - lat_remainder
        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top
    if lon_remainder:
        lon_pad = win_lon - lon_remainder
        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

    return padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back

def get_pad2d(input_resolution, window_size):
    """
    Args:
        input_resolution (tuple[int]): Lat, Lon
        window_size (tuple[int]): Lat, Lon

    Returns:
        padding (tuple[int]): (padding_left, padding_right, padding_top, padding_bottom)
    """
    input_resolution = [2] + list(input_resolution)
    window_size = [2] + list(window_size)
    padding = get_pad3d(input_resolution, window_size)
    return padding[: 4]


def time_to_features(timestamp, height: int, width: int, num_channels: int = 4) -> np.ndarray:
    dt = pd.Timestamp(str(timestamp))
    # 年相位：一年中的第几天 + 小时，归一化到 [0, 1]
    year_phase = (dt.dayofyear - 1 + dt.hour / 24.0) / 366.0

    features = []
    for i in range(num_channels):
        freq = 2 ** (i // 2)
        if i % 2 == 0:
            features.append(np.sin(2 * np.pi * freq * year_phase))
        else:
            features.append(np.cos(2 * np.pi * freq * year_phase))

    time_features = np.array(features, dtype=np.float32).reshape(-1, 1, 1)
    time_features = np.broadcast_to(time_features, (num_channels, height, width))
    return time_features


def time_to_features_batch(timestamps, height: int, width: int, device: torch.device, num_channels: int = 4) -> torch.Tensor:
    if isinstance(timestamps, (list, tuple)):
        ts_list = list(timestamps)
    else:
        ts_list = [str(t) for t in timestamps]

    features = [time_to_features(ts, height, width, num_channels) for ts in ts_list]
    features = np.stack(features, axis=0)
    return torch.from_numpy(features).to(device)




class PatchEmbedding(nn.Module):
    def __init__(self, img_size=(721, 1440), patch_size=(4, 4), in_chans=4, embed_dim=96, norm_layer=nn.LayerNorm):
        super().__init__()

        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.norm = norm_layer(embed_dim) if norm_layer else None
        self.patches_resolution = patches_resolution

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input size ({H}x{W}) doesn't match model ({self.img_size[0]}x{self.img_size[1]})"

        x = self.proj(x)

        if self.norm is not None:
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2)

        return x


class ResBlock(nn.Module):
    def __init__(self, num_groups, ch, dropout_rate: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, ch)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate and dropout_rate > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, ch)

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.dropout(h)
        h = self.norm2(self.conv2(h))
        return self.act(h + x)


class Downblock(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size = 3, stride = 2, padding = 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class Upblock(nn.Module):
    def __init__(self, in_chans, out_chans, out_size):
        super().__init__()
        self.size = out_size
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, size = tuple(self.size), mode = "bilinear"))



class UNetEncoder(nn.Module):
    def __init__(
        self,
        dim,
        num_groups,
        num_stages,
        output_reso,
        swin_depth,
        window_size,
        num_heads,
        using_checkpoints: bool = True,
        res_per_stage=None,
        dims=None,
        dropout_rate: float = 0.0,
        use_residual_blocks: bool = True,
        time_dim=None,
        source_dim=None,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.using_checkpoints = using_checkpoints
        self.use_residual_blocks = use_residual_blocks

        window_size = to_2tuple(window_size)

        base_h, base_w = output_reso
        self.stage_resolutions = []
        for i in range(num_stages):
            h_i = int(base_h // (2 ** i))
            w_i = int(base_w // (2 ** i))
            self.stage_resolutions.append((h_i, w_i))

        if dims is None:
            dims = [dim] * num_stages
        else:
            assert len(dims) == num_stages, "len(dims) must equal num_stages"
            dims = list(dims)
        self.dims = dims

        if res_per_stage is None:
            res_per_stage = [1] * num_stages
        elif isinstance(res_per_stage, int):
            res_per_stage = [res_per_stage] * num_stages
        else:
            assert len(res_per_stage) == num_stages, "len(res_per_stage) must equal num_stages"
        self.res_per_stage = res_per_stage

        if isinstance(swin_depth, int):
            depth_per_stage = [max(1, swin_depth // max(1, num_stages))] * num_stages
        else:
            assert len(swin_depth) == num_stages, "len(swin_depth) must equal num_stages"
            depth_per_stage = list(swin_depth)
        self.depth_per_stage = depth_per_stage

        self.res_blocks = nn.ModuleList()
        self.swin_stages = nn.ModuleList()
        self.down_blocks = nn.ModuleList()

        # Per-stage time projection: 1x1 conv projects time_dim → stage_dim
        self.stage_time_projs = nn.ModuleList()
        for i in range(num_stages):
            if time_dim is not None:
                self.stage_time_projs.append(nn.Conv2d(time_dim, self.dims[i], kernel_size=1))
            else:
                self.stage_time_projs.append(nn.Identity())

        # Per-stage source projection: 1x1 conv projects source_dim → stage_dim
        self.stage_source_projs = nn.ModuleList()
        for i in range(num_stages):
            if source_dim is not None:
                self.stage_source_projs.append(nn.Conv2d(source_dim, self.dims[i], kernel_size=1))
            else:
                self.stage_source_projs.append(nn.Identity())

        for i in range(num_stages):
            ch = self.dims[i]
            if self.use_residual_blocks:
                stage_res_blocks = nn.ModuleList(
                    [ResBlock(num_groups, ch, dropout_rate=dropout_rate) for _ in range(self.res_per_stage[i])]
                )
            else:
                stage_res_blocks = nn.ModuleList([])
            self.res_blocks.append(stage_res_blocks)

            input_reso = self.stage_resolutions[i]
            d_i = self.depth_per_stage[i]
            if d_i > 0:
                swin = SwinTransformerV2Stage(
                    dim=ch,
                    out_dim=ch,
                    window_size=window_size,
                    depth=d_i,
                    output_nchw=True,
                    input_resolution=input_reso,
                    num_heads=num_heads,
                )
                if using_checkpoints:
                    swin.grad_checkpointing = True
            else:
                swin = nn.Identity()
            self.swin_stages.append(swin)

            if i < num_stages - 1:
                self.down_blocks.append(Downblock(self.dims[i], self.dims[i + 1]))

    def forward(self, x, time_feats=None, source_feats=None):
        skips = []
        h = x
        for i in range(self.num_stages):
            # Time fusion: project time to current stage dim and spatial resolution
            if time_feats is not None:
                t_h, t_w = self.stage_resolutions[i]
                t = F.interpolate(time_feats, size=(t_h, t_w), mode='bilinear', align_corners=False)
                h = h + self.stage_time_projs[i](t)

            # Source fusion: project source to current stage dim and spatial resolution
            if source_feats is not None:
                s_h, s_w = self.stage_resolutions[i]
                s = F.interpolate(source_feats, size=(s_h, s_w), mode='bilinear', align_corners=False)
                h = h + self.stage_source_projs[i](s)

            ch = self.dims[i]
            for rb in self.res_blocks[i]:
                if self.using_checkpoints:
                    h = checkpoint.checkpoint(rb, h, use_reentrant=False)
                else:
                    h = rb(h)

            h_nhwc = h.permute(0, 2, 3, 1)
            h_nhwc = self.swin_stages[i](h_nhwc)
            h = h_nhwc.permute(0, 3, 1, 2)

            skips.append(h)

            if i < self.num_stages - 1:
                h = self.down_blocks[i](h)

        return h, skips


class UNetDecoder(nn.Module):
    def __init__(
        self,
        dim,
        num_groups,
        num_stages,
        output_reso,
        using_checkpoints: bool = True,
        dims=None,
        dropout_rate: float = 0.0,
        use_skip_connections: bool = True,
        use_residual_blocks: bool = True,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.using_checkpoints = using_checkpoints
        self.use_skip_connections = use_skip_connections
        self.use_residual_blocks = use_residual_blocks

        base_h, base_w = output_reso
        self.stage_resolutions = []
        for i in range(num_stages):
            h_i = int(base_h // (2 ** i))
            w_i = int(base_w // (2 ** i))
            self.stage_resolutions.append((h_i, w_i))

        if dims is None:
            dims = [dim] * num_stages
        else:
            assert len(dims) == num_stages, "len(dims) must equal num_stages"
            dims = list(dims)
        self.dims = dims

        self.up_blocks = nn.ModuleList()
        self.reduce_blocks = nn.ModuleList()
        self.res_blocks = nn.ModuleList()

        for idx in range(num_stages - 1):
            i = num_stages - 2 - idx
            out_size = self.stage_resolutions[i]
            in_ch = self.dims[i + 1]
            out_ch = self.dims[i]
            self.up_blocks.append(Upblock(in_ch, out_ch, out_size))

            reduce_in_ch = out_ch * 2 if self.use_skip_connections else out_ch
            reduce_layers = [
                nn.Conv2d(reduce_in_ch, out_ch, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups, out_ch),
                nn.SiLU(),
            ]
            if dropout_rate and dropout_rate > 0:
                reduce_layers.append(nn.Dropout2d(dropout_rate))
            self.reduce_blocks.append(nn.Sequential(*reduce_layers))
            if self.use_residual_blocks:
                self.res_blocks.append(ResBlock(num_groups, out_ch, dropout_rate=dropout_rate))
            else:
                self.res_blocks.append(nn.Identity())

    def forward(self, x, skips):
        h = x
        for idx, (up, reduce, res) in enumerate(zip(self.up_blocks, self.reduce_blocks, self.res_blocks)):
            i = self.num_stages - 2 - idx
            h = up(h)
            if self.use_skip_connections and skips is not None:
                h = torch.cat([h, skips[i]], dim=1)
            h = reduce(h)
            if self.using_checkpoints:
                h = checkpoint.checkpoint(res, h, use_reentrant=False)
            else:
                h = res(h)
        return h


class UNet(nn.Module):
    def __init__(
        self,
        dim,
        num_groups,
        num_stages,
        output_reso,
        swin_depth,
        window_size,
        num_heads,
        using_checkpoints: bool = True,
        res_per_stage=None,
        dims=None,
        using_kl: bool = False,
        dropout_rate: float = 0.0,
        use_skip_connections: bool = True,
        use_residual_blocks: bool = True,
        time_dim=None,
        source_dim=None,
        using_dann: bool = False,
        dann_hidden_dim: int = 256,
        dann_num_domains: int = 2,
        **kwargs,
    ):
        super().__init__()
        window_size = to_2tuple(window_size)

        self.using_kl = using_kl
        self.using_dann = using_dann

        if dims is None:
            dims = [dim] * num_stages
        else:
            assert len(dims) == num_stages, "len(dims) must equal num_stages"
            dims = list(dims)

        self.encoder = UNetEncoder(
            dim,
            num_groups,
            num_stages,
            output_reso,
            swin_depth,
            window_size,
            num_heads,
            using_checkpoints=using_checkpoints,
            res_per_stage=res_per_stage,
            dims=dims,
            dropout_rate=dropout_rate,
            use_residual_blocks=use_residual_blocks,
            time_dim=time_dim,
            source_dim=source_dim,
        )

        self.decoder = UNetDecoder(
            dim,
            num_groups,
            num_stages,
            output_reso,
            using_checkpoints=using_checkpoints,
            dims=dims,
            dropout_rate=dropout_rate,
            use_skip_connections=use_skip_connections,
            use_residual_blocks=use_residual_blocks,
        )

        if self.using_kl:
            bottleneck_ch = dims[-1]
            self.mu_head = nn.Conv2d(bottleneck_ch, bottleneck_ch, kernel_size=3, padding=1)
            self.logvar_head = nn.Conv2d(bottleneck_ch, bottleneck_ch, kernel_size=3, padding=1)

        # DANN 域分类器：挂在 encoder bottleneck 之后，判断 embedding 来自哪个源域
        if using_dann:
            bottleneck_ch = dims[-1]
            self.domain_classifier = DomainClassifier(
                in_dim=bottleneck_ch,
                hidden_dim=dann_hidden_dim,
                num_domains=dann_num_domains,
            )
        else:
            self.domain_classifier = None

    def forward(self, x, time_feats=None, source_feats=None, domains=None, grl_lambda=1.0):
        bottleneck, skips = self.encoder(x, time_feats=time_feats, source_feats=source_feats)

        # DANN: 从 bottleneck 池化后经域分类器预测源域
        domain_logits = None
        if self.domain_classifier is not None and domains is not None:
            z_pool = bottleneck.mean(dim=[2, 3])
            domain_logits = self.domain_classifier(z_pool, lambda_=grl_lambda)

        if self.using_kl:
            mu = self.mu_head(bottleneck)
            log_var = self.logvar_head(bottleneck)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + eps * std
            out = self.decoder(z, skips)
            if domain_logits is not None:
                return out, mu, log_var, domain_logits
            return out, mu, log_var
        else:
            out = self.decoder(bottleneck, skips)
            if domain_logits is not None:
                return out, domain_logits
            return out



class PatchHead(nn.Module):
    def __init__(self, embed_dim, out_chans, patch_size=(4,4)):
        super().__init__()

        self.patch_size = patch_size
        self.out_chans = out_chans
        self.head = nn.Linear(embed_dim, out_chans * patch_size[0] * patch_size[1])


    def forward(self, x):
        B, C, H, W = x.shape

        feat_h, feat_w = H, W

        x = x.flatten(2).transpose(1, 2)

        x = self.head(x)
        x = rearrange(
            x,
            'n (h w) (p1 p2 c) -> n c (h p1) (w p2)',
            h=feat_h,
            w=feat_w,
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            c=self.out_chans
        )

        return x

class A2E(nn.Module):
    """Any-to-ERA5: Multi-source weather field translation model.

    Takes input from multiple source domains (GFS, HRES, CMA, etc.) and maps to
    ERA5-equivalent fields. Uses a learned source-type embedding (analogous to
    time embedding) concatenated as additional input channels, so a single shared
    encoder/decoder can learn source-specific bias, resolution, and error patterns.
    """

    def __init__(
        self,
        img_size=(721, 1440),
        patch_size=(4, 4),
        in_chans=10,
        out_chans = None,
        embed_dim=1536,
        num_groups=32,
        num_heads=8,
        num_stages=3,
        window_size=9,
        depth = 12,
        latent_dim = 1536,
        using_checkpoints = True,
        using_time_embedding = False,
        using_source_embedding = False,
        num_sources = 1,
        source_embed_dim = None,
        res_per_stage = [1, 2, 4],
        channels=None,
        using_kl: bool = False,
        dropout_rate: float = 0.0,
        use_skip_connections: bool = True,
        use_residual_blocks: bool = True,
        using_dann: bool = False,
        dann_hidden_dim: int = 256,
        **kwargs

    ):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = in_chans if out_chans is None else out_chans
        self.patch_size = patch_size
        self.img_size = img_size
        self.using_checkpoints = using_checkpoints
        self.using_time_embedding = using_time_embedding

        self.using_source_embedding = using_source_embedding
        self.num_sources = num_sources
        if source_embed_dim is None:
            source_embed_dim = in_chans
        self.source_embed_dim = source_embed_dim
        self.using_kl = using_kl
        self.using_dann = using_dann

        self.time_channels = in_chans if using_time_embedding else 0

        input_resolution = int(img_size[0] / patch_size[0]), int(img_size[1] / patch_size[1])

        if channels is None:
            dims = [embed_dim] * num_stages
        else:
            assert len(channels) == num_stages, "len(channels) must equal num_stages"
            dims = list(channels)
        self.dims = dims

        # Learned source-type embedding: one vector per source domain,
        # expanded to (B, source_embed_dim, H, W) and fused via 1x1 conv projection.
        if using_source_embedding and num_sources > 0:
            self.source_embed = nn.Embedding(num_sources, source_embed_dim)
            self.input_source_proj = nn.Conv2d(source_embed_dim, in_chans, kernel_size=1)
        else:
            self.source_embed = None
            self.input_source_proj = None

        # Time projection: fuse time features with image via 1x1 conv (no concat)
        if using_time_embedding:
            self.input_time_proj = nn.Conv2d(in_chans, in_chans, kernel_size=1)
        else:
            self.input_time_proj = None

        # Patch embedding: data only (time & source fused via projection, not concat)
        total_in_chans = in_chans
        self.patch_emb = PatchEmbedding(img_size, patch_size, total_in_chans, self.dims[0])

        encoder_time_dim = self.time_channels if using_time_embedding else None
        encoder_source_dim = source_embed_dim if using_source_embedding else None
        self.mid_layer = UNet(
            self.dims[0],
            num_groups,
            num_stages,
            input_resolution,
            depth,
            window_size,
            num_heads,
            using_checkpoints,
            res_per_stage,
            dims=self.dims,
            using_kl=self.using_kl,
            dropout_rate=dropout_rate,
            use_skip_connections=use_skip_connections,
            use_residual_blocks=use_residual_blocks,
            time_dim=encoder_time_dim,
            source_dim=encoder_source_dim,
            using_dann=using_dann,
            dann_hidden_dim=dann_hidden_dim,
            dann_num_domains=num_sources,
        )

        self.patch_head = PatchHead(self.dims[0], self.out_chans, patch_size)


    def forward(self, x, times=None, domains=None, source_idx=None, grl_lambda=1.0):
        """Forward pass.

        Args:
            x: Input tensor [B, in_chans, H, W]
            times: Batch of time strings, used for time embedding [B]
            domains: Source domain indices [B], dtype long.
            source_idx: Backward-compatible alias for domains.
            grl_lambda: GRL 梯度反转系数，仅 using_dann=True 时生效。
        """
        B, C, H, W = x.shape

        time_feats = None
        if self.using_time_embedding and times is not None:
            time_feats = time_to_features_batch(times, H, W, x.device, num_channels=self.time_channels)
            x = x + self.input_time_proj(time_feats)

        if domains is None:
            domains = source_idx

        source_feats = None
        if self.using_source_embedding and domains is not None and self.source_embed is not None:
            source_emb = self.source_embed(domains)
            source_feats = source_emb[:, :, None, None].expand(B, -1, H, W)
            x = x + self.input_source_proj(source_feats)

        if self.using_checkpoints:
            x_patch = checkpoint.checkpoint(self.patch_emb, x, use_reentrant=False)
        else:
            x_patch = self.patch_emb(x)

        result = self.mid_layer(
            x_patch,
            time_feats=time_feats,
            source_feats=source_feats,
            domains=domains,
            grl_lambda=grl_lambda,
        )

        # 解包 UNet 返回值
        if self.using_kl and self.using_dann:
            mid_out, mu, log_var, domain_logits = result
        elif self.using_kl:
            mid_out, mu, log_var = result
            domain_logits = None
        elif self.using_dann:
            mid_out, domain_logits = result
        else:
            mid_out = result
            domain_logits = None

        if self.using_checkpoints:
            x = checkpoint.checkpoint(self.patch_head, mid_out, use_reentrant=False)
        else:
            x = self.patch_head(mid_out)

        x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=False)
        if self.using_kl:
            return x, mu, log_var, domain_logits
        elif self.using_dann:
            return x, domain_logits
        else:
            return x

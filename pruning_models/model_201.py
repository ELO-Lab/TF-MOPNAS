import torch.nn as nn

def get_model_from_arch_str(keep_mask, num_classes, use_bn=True, init_channels=16):
    net = NAS201Model(
        num_classes=num_classes,
        use_bn=use_bn,
        keep_mask=keep_mask,
        stem_ch=init_channels,
    )
    return net

class ReLUConvBN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        affine,
        track_running_stats=True,
        use_bn=True,
        name="ReLUConvBN",
    ):
        super(ReLUConvBN, self).__init__()
        self.name = name
        if use_bn:
            self.op = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    bias=not affine,
                ),
                nn.BatchNorm2d(
                    out_channels, affine=affine, track_running_stats=track_running_stats
                ),
            )
        else:
            self.op = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    bias=not affine,
                ),
            )

    def forward(self, x):
        return self.op(x)

class Identity(nn.Module):
    def __init__(self, name="Identity"):
        self.name = name
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Zero(nn.Module):
    def __init__(self, stride, name="Zero"):
        self.name = name
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, :: self.stride, :: self.stride].mul(0.0)

class POOLING(nn.Module):
    def __init__(self, kernel_size, stride, padding, name="POOLING"):
        super(POOLING, self).__init__()
        self.name = name
        self.avgpool = nn.AvgPool2d(
            kernel_size=kernel_size, stride=1, padding=1, count_include_pad=False
        )

    def forward(self, x):
        return self.avgpool(x)

class stem(nn.Module):
    def __init__(self, out_channels, use_bn=True):
        super(stem, self).__init__()
        if use_bn:
            self.net = nn.Sequential(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                )
            )

    def forward(self, x):
        return self.net(x)

class reduction(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(reduction, self).__init__()
        self.residual = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
        )

        self.conv_a = ReLUConvBN(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            affine=True,
            track_running_stats=True,
        )
        self.conv_b = ReLUConvBN(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            affine=True,
            track_running_stats=True,
        )

    def forward(self, x):
        basicblock = self.conv_a(x)
        basicblock = self.conv_b(basicblock)
        residual = self.residual(x)
        return residual + basicblock

class top(nn.Module):
    def __init__(self, in_dims, num_classes, use_bn=True):
        super(top, self).__init__()
        if use_bn:
            self.lastact = nn.Sequential(nn.BatchNorm2d(in_dims, track_running_stats=True), nn.ReLU(inplace=True))
        else:
            self.lastact = nn.ReLU(inplace=True)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_dims, num_classes)

    def forward(self, x):
        x = self.lastact(x)
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits

class SearchCell(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        affine,
        track_running_stats,
        use_bn=True,
        num_nodes=4,
        keep_mask=None,
    ):
        super(SearchCell, self).__init__()
        self.num_nodes = num_nodes
        self.options = nn.ModuleList()
        for curr_node in range(self.num_nodes - 1):
            for prev_node in range(curr_node + 1):
                for _op_name in OPS.keys():
                    op = OPS[_op_name](
                        in_channels,
                        out_channels,
                        stride,
                        affine,
                        track_running_stats,
                        use_bn,
                    )
                    self.options.append(op)

        if keep_mask is not None:
            self.keep_mask = keep_mask
        else:
            self.keep_mask = [True] * len(self.options)

    def forward(self, x):
        outs = [x]

        idx = 0
        for curr_node in range(self.num_nodes - 1):
            edges_in = []
            for prev_node in range(curr_node + 1):  # n-1 prev nodes
                for op_idx in range(len(OPS.keys())):
                    if self.keep_mask[idx]:
                        edges_in.append(self.options[idx](outs[prev_node]))
                    idx += 1
            node_output = sum(edges_in)
            outs.append(node_output)

        return outs[-1]

OPS = {
    "none": lambda in_channels, out_channels, stride, affine, track_running_stats, use_bn: Zero(
        stride, name="none"
    ),
    "skip_connect": lambda in_channels, out_channels, stride, affine, track_running_stats, use_bn: Identity(
        name="skip_connect"
    ),
    "nor_conv_1x1": lambda in_channels, out_channels, stride, affine, track_running_stats, use_bn: ReLUConvBN(
        in_channels,
        out_channels,
        1,
        1,
        0,
        1,
        affine,
        track_running_stats,
        use_bn,
        name="nor_conv_1x1",
    ),
    "nor_conv_3x3": lambda in_channels, out_channels, stride, affine, track_running_stats, use_bn: ReLUConvBN(
        in_channels,
        out_channels,
        3,
        1,
        1,
        1,
        affine,
        track_running_stats,
        use_bn,
        name="nor_conv_3x3",
    ),
    "avg_pool_3x3": lambda in_channels, out_channels, stride, affine, track_running_stats, use_bn: POOLING(
        3, 1, 1, name="avg_pool_3x3"
    )
}

class NAS201Model(nn.Module):
    def __init__(self, num_classes, use_bn=True, keep_mask=None, stem_ch=16):
        super(NAS201Model, self).__init__()
        self.num_classes = num_classes
        self.use_bn = use_bn

        self.stem = stem(out_channels=stem_ch, use_bn=use_bn)
        self.stack_cell1 = nn.Sequential(
            *[
                SearchCell(
                    in_channels=stem_ch,
                    out_channels=stem_ch,
                    stride=1,
                    affine=False,
                    track_running_stats=True,
                    use_bn=use_bn,
                    keep_mask=keep_mask,
                )
                for i in range(5)
            ]
        )
        self.reduction1 = reduction(in_channels=stem_ch, out_channels=stem_ch * 2)
        self.stack_cell2 = nn.Sequential(
            *[
                SearchCell(
                    in_channels=stem_ch * 2,
                    out_channels=stem_ch * 2,
                    stride=1,
                    affine=False,
                    track_running_stats=True,
                    use_bn=use_bn,
                    keep_mask=keep_mask,
                )
                for i in range(5)
            ]
        )
        self.reduction2 = reduction(in_channels=stem_ch * 2, out_channels=stem_ch * 4)
        self.stack_cell3 = nn.Sequential(
            *[
                SearchCell(
                    in_channels=stem_ch * 4,
                    out_channels=stem_ch * 4,
                    stride=1,
                    affine=False,
                    track_running_stats=True,
                    use_bn=use_bn,
                    keep_mask=keep_mask,
                )
                for i in range(5)
            ]
        )
        self.top = top(in_dims=stem_ch * 4, num_classes=num_classes, use_bn=use_bn)

    def forward(self, x):
        x = self.stem(x)

        x = self.stack_cell1(x)
        x = self.reduction1(x)

        x = self.stack_cell2(x)
        x = self.reduction2(x)

        x = self.stack_cell3(x)

        x = self.top(x)
        return x

    def forward_pre_GAP(self, x):
        x = self.stem(x)

        x = self.stack_cell1(x)
        x = self.reduction1(x)

        x = self.stack_cell2(x)
        x = self.reduction2(x)

        x = self.stack_cell3(x)

        x = self.top.lastact(x)
        return x
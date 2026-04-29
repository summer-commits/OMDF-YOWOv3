from .CFAM import CFAMFusion
from .SE import SEFusion
from .Simple import SimpleFusion
from .MultiHead import MultiHeadFusion
from .Channel import ChannelFusion
from .Spatial import SpatialFusion
from .CBAM import CBAMFusion
from .LKA import LKAFusion
from .docf import DOCFFusion


def build_fusion_block(out_channels_2D,
                       out_channels_3D,
                       inter_channels_fusion,
                       mode,
                       fusion_block,
                       lastdimension,
                       config=None):
    if fusion_block == 'CFAM':
        return CFAMFusion(out_channels_2D, out_channels_3D, inter_channels_fusion, mode)
    elif fusion_block == 'SE':
        return SEFusion(out_channels_2D, out_channels_3D, inter_channels_fusion, mode)
    elif fusion_block == 'Simple':
        return SimpleFusion(out_channels_2D, out_channels_3D, inter_channels_fusion, mode)
    elif fusion_block == 'MultiHead':
        return MultiHeadFusion(out_channels_2D, out_channels_3D, inter_channels_fusion, lastdimension, mode, h=1)
    elif fusion_block == 'Channel':
        return ChannelFusion(out_channels_2D, out_channels_3D, inter_channels_fusion, mode)
    elif fusion_block == 'Spatial':
        return SpatialFusion(out_channels_2D, out_channels_3D, inter_channels_fusion, mode)
    elif fusion_block == 'CBAM':
        return CBAMFusion(out_channels_2D, out_channels_3D, inter_channels_fusion, mode)
    elif fusion_block == 'LKA':
        return LKAFusion(out_channels_2D, out_channels_3D, inter_channels_fusion, mode)
    elif fusion_block == 'DOCF':
        # lastdimension: [dims_box, dims_cls]
        # dims_box/dims_cls: [(H3,W3), (H4,W4), (H5,W5)]
        dims_box, dims_cls = lastdimension
        assert len(dims_box) == len(dims_cls) == 3, "Expect 3 feature levels P3/P4/P5"
        for i in range(3):
            assert dims_box[i] == dims_cls[i], f"Box/Cls spatial mismatch at level {i}: {dims_box[i]} vs {dims_cls[i]}"

        ic = (inter_channels_fusion[0] if isinstance(inter_channels_fusion, (list, tuple))
              else int(inter_channels_fusion))
        hw_info = [(int(x[0]), int(x[1])) for x in dims_box]

        cfg = config or {}
        fusion_cfg = (cfg.get('FUSION', {}) or {})
        docf_cfg = (fusion_cfg.get('DOCF', {}) or {})

        num_layers = int(docf_cfg.get('num_layers', 2))
        gating = bool(docf_cfg.get('gating', True))
        upsample = str(docf_cfg.get('upsample', 'carafe')).lower()
        use_carafe = (upsample == 'carafe')

        inner_docf_cfg = docf_cfg.get('DOCF', {}) or docf_cfg.get('docf', {}) or docf_cfg

        return DOCFFusion(
            out_channels_2D,
            out_channels_3D,
            ic,
            mode=mode,
            num_layers=num_layers,
            use_carafe=use_carafe,
            gating=gating,
            hw_info=hw_info,
            docf_cfg=inner_docf_cfg
        )
    else:
        raise ValueError(f"Unknown fusion block: {fusion_block}")

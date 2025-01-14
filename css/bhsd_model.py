# from monai.networks.nets import SwinUNETR
# from css.swin_unetr_css_merging import SwinUNETR
from css.swin_unetr_css_merging_skip import SwinUNETR as SwinUNETR
from css.swin_unetr_css_merging_skipV2 import SwinUNETR as SwinUNETR_css_merging
from css.swin_unetr_css_merging_skipV3 import SwinUNETR as SwinUNETR_css_MultiScaleMerging
from css.swin_unetr_css_merging_skipV3_AddEncForX3 import SwinUNETR as SwinUNETR_css_merging_skipV3_AddEncForX3
from css.swin_unetr_css_merging_skipV3_maxpoolskip import SwinUNETR as SwinUNETR_css_merging_skipV3_maxpoolskip
from css.swin_unetr_css_merging_skipV3_avgpoolskip import SwinUNETR as SwinUNETR_css_merging_skipV3_avgpoolskip
from css.swin_unetr_css_merging_skipV3_maxavgpoolskip import SwinUNETR as SwinUNETR_css_merging_skipV3_maxavgpoolskip
from css.swin_unetr_css_merging_skipV3_convskip import SwinUNETR as SwinUNETR_css_merging_skipV3_convskip
from css.swin_unetr_css_merging_skipV3_convskip_AddM0 import SwinUNETR as SwinUNETR_css_merging_skipV3_convskip_AddM0
from css.swin_unetr_css_merging_skipV4_imageconvMerging import SwinUNETR as SwinUNETR_css_merging_imageconvMerging
from css.swin_unetr_css_merging_skipV4_imageconvInSkip import SwinUNETR as SwinUNETR_css_merging_imageconvInSkip
from css.swin_unetr_css_merging_skipV4 import SwinUNETR as SwinUNETR_2_3MultiScaleMerging
import torch
import os
import json
def css_model(args):
    if args.model == 'swin_unetr':
        # ******************************************Create Swin UNETR model*******************************************
        print("Using swin_unetr!")
        model = SwinUNETR(
            img_size=args.ref_window,
            in_channels=1,
            out_channels=6,
            feature_size=48,
            use_checkpoint=True,
            # use_v2=True,
            use_ln = args.use_ln,
            merging_type=args.merging_type,
            css_skip=args.css_skip,
            use_1x1_conv_for_skip=args.use_1x1_conv_for_skip,
            use_css_skip_m4 = args.use_css_skip_m4,
            use_css_skip_m1V2 = args.use_css_skip_m1V2
        ).to(args.device)
        if args.test:
            assert os.path.isfile(args.ref_weight), "weight path is not a file"
            print(f"weight_path: {args.ref_weight}")
            model.load_state_dict(torch.load(args.ref_weight))
            return model
        weight = torch.load("css/model_swinvit.pt")
        model.load_from(weights=weight)
        print("Using pretrained self-supervied Swin UNETR backbone weights !")
        return model
    
    elif args.model == 'swin_unetr_css_merging':
        print("Using swin_unetr_css_merging!")
        # ******************************************Create Swin UNETR model*******************************************
        model = SwinUNETR_css_merging(
            img_size=args.ref_window,
            in_channels=1,
            out_channels=6,
            feature_size=48,
            use_checkpoint=True,
            use_ln = args.use_ln,
            # use_v2=True,
            merging_type=args.merging_type,
            css_skip=args.css_skip,
            use_1x1_conv_for_skip=args.use_1x1_conv_for_skip,
            use_css_skip_m4 = args.use_css_skip_m4,
            use_css_skip_m1V2 = args.use_css_skip_m1V2
        ).to(args.device)
        if args.test:
            assert os.path.isfile(args.ref_weight), "weight path is not a file"
            print(f"weight_path: {args.ref_weight}")
            model.load_state_dict(torch.load(args.ref_weight))
            return model
        weight = torch.load("css/model_swinvit.pt")
        model.load_from(weights=weight)
        print("Using pretrained self-supervied Swin UNETR backbone weights !")
        return model
    
    elif args.model == 'swin_unetr_css_MultiScaleMerging':
        print("Using swin_unetr_css_MultiScaleMerging!")
        # ******************************************Create Swin UNETR model*******************************************
        model = SwinUNETR_css_MultiScaleMerging(
            img_size=args.ref_window,
            in_channels=1,
            out_channels=6,
            feature_size=48,
            use_checkpoint=True,
            use_ln = args.use_ln,
            # use_v2=True,
            merging_type=args.merging_type,
            css_skip=args.css_skip,
            use_1x1_conv_for_skip=args.use_1x1_conv_for_skip,
            use_css_skip_m4 = args.use_css_skip_m4,
            use_css_skip_m1V2 = args.use_css_skip_m1V2
        ).to(args.device)
        if args.test:
            assert os.path.isfile(args.ref_weight), "weight path is not a file"
            print(f"weight_path: {args.ref_weight}")
            model.load_state_dict(torch.load(args.ref_weight))
            return model
        weight = torch.load("css/model_swinvit.pt")
        model.load_from(weights=weight)
        print("Using pretrained self-supervied Swin UNETR backbone weights !")
        return model
    
    elif args.model == 'SwinUNETR_css_merging_skipV3_AddEncForX3':
        print("Using SwinUNETR_css_merging_skipV3_AddEncForX3!")
        # ******************************************Create Swin UNETR model*******************************************
        model = SwinUNETR_css_merging_skipV3_AddEncForX3(
            img_size=args.ref_window,
            in_channels=1,
            out_channels=6,
            feature_size=48,
            use_checkpoint=True,
            use_ln = args.use_ln,
            # use_v2=True,
            merging_type=args.merging_type,
            css_skip=args.css_skip,
            use_1x1_conv_for_skip=args.use_1x1_conv_for_skip,
            use_css_skip_m4 = args.use_css_skip_m4,
            use_css_skip_m1V2 = args.use_css_skip_m1V2
        ).to(args.device)
        if args.test:
            assert os.path.isfile(args.ref_weight), "weight path is not a file"
            print(f"weight_path: {args.ref_weight}")
            model.load_state_dict(torch.load(args.ref_weight))
            return model
        weight = torch.load("css/model_swinvit.pt")
        model.load_from(weights=weight)
        print("Using pretrained self-supervied Swin UNETR backbone weights !")
        return model
    
    elif args.model == 'SwinUNETR_css_merging_skipV3_maxpoolskip':
        print("Using SwinUNETR_css_merging_skipV3_maxpoolskip!")
        # ******************************************Create Swin UNETR model*******************************************
        model = SwinUNETR_css_merging_skipV3_maxpoolskip(
            img_size=args.ref_window,
            in_channels=1,
            out_channels=6,
            feature_size=48,
            use_checkpoint=True,
            use_ln = args.use_ln,
            # use_v2=True,
            merging_type=args.merging_type,
            css_skip=args.css_skip,
            use_1x1_conv_for_skip=args.use_1x1_conv_for_skip,
            use_dec_change_C_in_css_skip=args.use_dec_change_C_in_css_skip,
            use_css_skip_m4 = args.use_css_skip_m4,
            use_css_skip_m1V2 = args.use_css_skip_m1V2
        ).to(args.device)
        if args.test:
            assert os.path.isfile(args.ref_weight), "weight path is not a file"
            print(f"weight_path: {args.ref_weight}")
            model.load_state_dict(torch.load(args.ref_weight))
            return model
        weight = torch.load("css/model_swinvit.pt")
        model.load_from(weights=weight)
        print("Using pretrained self-supervied Swin UNETR backbone weights !")
        return model
    
    elif args.model == 'SwinUNETR_css_merging_skipV3_avgpoolskip':
        print("Using SwinUNETR_css_merging_skipV3_avgpoolskip!")
        # ******************************************Create Swin UNETR model*******************************************
        model = SwinUNETR_css_merging_skipV3_avgpoolskip(
            img_size=args.ref_window,
            in_channels=1,
            out_channels=6,
            feature_size=48,
            use_checkpoint=True,
            use_ln = args.use_ln,
            # use_v2=True,
            merging_type=args.merging_type,
            css_skip=args.css_skip,
            use_1x1_conv_for_skip=args.use_1x1_conv_for_skip,
            use_dec_change_C_in_css_skip=args.use_dec_change_C_in_css_skip,
            use_css_skip_m4 = args.use_css_skip_m4,
            use_css_skip_m1V2 = args.use_css_skip_m1V2
        ).to(args.device)
        if args.test:
            assert os.path.isfile(args.ref_weight), "weight path is not a file"
            print(f"weight_path: {args.ref_weight}")
            model.load_state_dict(torch.load(args.ref_weight))
            return model
        weight = torch.load("css/model_swinvit.pt")
        model.load_from(weights=weight)
        print("Using pretrained self-supervied Swin UNETR backbone weights !")
        return model
    
    elif args.model == 'SwinUNETR_css_merging_skipV3_maxavgpoolskip':
        print("Using SwinUNETR_css_merging_skipV3_maxavgpoolskip!")
        # ******************************************Create Swin UNETR model*******************************************
        model = SwinUNETR_css_merging_skipV3_maxavgpoolskip(
            img_size=args.ref_window,
            in_channels=1,
            out_channels=6,
            feature_size=48,
            use_checkpoint=True,
            use_ln = args.use_ln,
            # use_v2=True,
            merging_type=args.merging_type,
            css_skip=args.css_skip,
            use_1x1_conv_for_skip=args.use_1x1_conv_for_skip,
            use_dec_change_C_in_css_skip=args.use_dec_change_C_in_css_skip,
            use_css_skip_m4 = args.use_css_skip_m4,
            use_css_skip_m1V2 = args.use_css_skip_m1V2
        ).to(args.device)
        if args.test:
            assert os.path.isfile(args.ref_weight), "weight path is not a file"
            print(f"weight_path: {args.ref_weight}")
            model.load_state_dict(torch.load(args.ref_weight))
            return model
        weight = torch.load("css/model_swinvit.pt")
        model.load_from(weights=weight)
        print("Using pretrained self-supervied Swin UNETR backbone weights !")
        return model
    
    elif args.model == 'SwinUNETR_css_merging_skipV3_convskip':
        print("Using SwinUNETR_css_merging_skipV3_convskip!")
        # ******************************************Create Swin UNETR model*******************************************
        model = SwinUNETR_css_merging_skipV3_convskip(
            img_size=args.ref_window,
            in_channels=1,
            out_channels=6,
            feature_size=48,
            use_checkpoint=True,
            use_ln = args.use_ln,
            # use_v2=True,
            merging_type=args.merging_type,
            css_skip=args.css_skip,
            use_1x1_conv_for_skip=args.use_1x1_conv_for_skip,
            use_dec_change_C_in_css_skip=args.use_dec_change_C_in_css_skip,
            use_css_skip_m4 = args.use_css_skip_m4,
            use_css_skip_m1V2 = args.use_css_skip_m1V2
        ).to(args.device)
        if args.test:
            assert os.path.isfile(args.ref_weight), "weight path is not a file"
            print(f"weight_path: {args.ref_weight}")
            model.load_state_dict(torch.load(args.ref_weight))
            return model
        weight = torch.load("css/model_swinvit.pt")
        model.load_from(weights=weight)
        print("Using pretrained self-supervied Swin UNETR backbone weights !")
        return model
    
    elif args.model == 'SwinUNETR_css_merging_skipV3_convskip_AddM0':
        print("Using SwinUNETR_css_merging_skipV3_convskip_AddM0!")
        # ******************************************Create Swin UNETR model*******************************************
        model = SwinUNETR_css_merging_skipV3_convskip_AddM0(
            img_size=args.ref_window,
            in_channels=1,
            out_channels=6,
            feature_size=48,
            use_checkpoint=True,
            use_ln = args.use_ln,
            # use_v2=True,
            merging_type=args.merging_type,
            css_skip=args.css_skip,
            use_1x1_conv_for_skip=args.use_1x1_conv_for_skip,
            use_dec_change_C_in_css_skip=args.use_dec_change_C_in_css_skip,
            use_css_skip_m4 = args.use_css_skip_m4,
            use_css_skip_m1V2 = args.use_css_skip_m1V2
        ).to(args.device)
        if args.test:
            assert os.path.isfile(args.ref_weight), "weight path is not a file"
            print(f"weight_path: {args.ref_weight}")
            model.load_state_dict(torch.load(args.ref_weight))
            return model
        weight = torch.load("css/model_swinvit.pt")
        model.load_from(weights=weight)
        print("Using pretrained self-supervied Swin UNETR backbone weights !")
        return model

    elif args.model == 'SwinUNETR_css_merging_imageconvMerging':
        print("Using SwinUNETR_css_merging_imageconvMerging!")
        # ******************************************Create Swin UNETR model*******************************************
        model = SwinUNETR_css_merging_imageconvMerging(
            img_size=args.ref_window,
            in_channels=1,
            out_channels=6,
            feature_size=48,
            use_checkpoint=True,
            use_ln = args.use_ln,
            # use_v2=True,
            merging_type=args.merging_type,
            css_skip=args.css_skip,
            use_1x1_conv_for_skip=args.use_1x1_conv_for_skip,
            use_dec_change_C_in_css_skip=args.use_dec_change_C_in_css_skip,
            use_css_skip_m4 = args.use_css_skip_m4,
            use_css_skip_m1V2 = args.use_css_skip_m1V2
        ).to(args.device)
        if args.test:
            assert os.path.isfile(args.ref_weight), "weight path is not a file"
            print(f"weight_path: {args.ref_weight}")
            model.load_state_dict(torch.load(args.ref_weight))
            return model
        weight = torch.load("css/model_swinvit.pt")
        model.load_from(weights=weight)
        print("Using pretrained self-supervied Swin UNETR backbone weights !")
        return model
    
    elif args.model == 'SwinUNETR_css_merging_imageconvInSkip':
        print("Using SwinUNETR_css_merging_imageconvInSkip!")
        # ******************************************Create Swin UNETR model*******************************************
        model = SwinUNETR_css_merging_imageconvInSkip(
            img_size=args.ref_window,
            in_channels=1,
            out_channels=6,
            feature_size=48,
            use_checkpoint=True,
            use_ln = args.use_ln,
            # use_v2=True,
            merging_type=args.merging_type,
            css_skip=args.css_skip,
            use_1x1_conv_for_skip=args.use_1x1_conv_for_skip,
            use_dec_change_C_in_css_skip=args.use_dec_change_C_in_css_skip,
            use_css_skip_m4 = args.use_css_skip_m4,
            use_css_skip_m1V2 = args.use_css_skip_m1V2
        ).to(args.device)
        if args.test:
            assert os.path.isfile(args.ref_weight), "weight path is not a file"
            print(f"weight_path: {args.ref_weight}")
            model.load_state_dict(torch.load(args.ref_weight))
            return model
        weight = torch.load("css/model_swinvit.pt")
        model.load_from(weights=weight)
        print("Using pretrained self-supervied Swin UNETR backbone weights !")
        return model


    elif args.model == 'SwinUNETR_2_3MultiScaleMerging':
        print("Using SwinUNETR_2_3MultiScaleMerging!")
        # ******************************************Create Swin UNETR model*******************************************
        model = SwinUNETR_2_3MultiScaleMerging(
            img_size=args.ref_window,
            in_channels=1,
            out_channels=6,
            feature_size=48,
            use_checkpoint=True,
            use_ln = args.use_ln,
            # use_v2=True,
            merging_type=args.merging_type,
            css_skip=args.css_skip,
            use_1x1_conv_for_skip=args.use_1x1_conv_for_skip,
            # use_dec_change_C_in_css_skip=args.use_dec_change_C_in_css_skip,
            use_css_skip_m4 = args.use_css_skip_m4,
            use_css_skip_m1V2 = args.use_css_skip_m1V2
        ).to(args.device)
        if args.test:
            assert os.path.isfile(args.ref_weight), "weight path is not a file"
            print(f"weight_path: {args.ref_weight}")
            model.load_state_dict(torch.load(args.ref_weight))
            return model
        weight = torch.load("css/model_swinvit.pt")
        model.load_from(weights=weight)
        print("Using pretrained self-supervied Swin UNETR backbone weights !")
        return model

    else:
        raise ValueError(f'暂不支持{args.model}~')
from Unet_model import*



# 定义模式对应的函数字典
mode_functions = {

    "ThreeBand": {
        "bands": 3,
        "model": unet_three_band,
    },
    "ThreeBand_FPN": {
        "bands": 3,
        "model": unet_three_band_with_fpn,
    },
    "ThreeBand_attention": {
        "bands": 3,
        "model": unet_three_band_attention,
    },
    "ThreeBand_FPN_attention": {
        "bands": 3,
        "model": unet_three_band_with_fpn_CBAM_Encoder,
    },
    "TwoBand": {
        "bands": 2,
        "model": unet_two_band,
    },
    "TwoBand_attention": {
        "bands": 2,
        "model": unet_two_band_attention,
    },

    "TenBand":{
        "bands": 10,
        "model": unet_ten_band,
    }
}
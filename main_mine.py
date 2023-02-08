from Train_mine import train, eval


def main(model_config = None):
    modelConfig = {
        # "state": "train", # or eval//
        "state": "eval",  # or eval

        # "epoch": 200,
        "epoch": 10,
        "batch_size": 32,
        "T": 100,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_T": 0.02,
        "img_size": 128,
        "grad_clip": 1.,
        "device": "cuda:1",
        "training_load_weight": None,
        "save_weight_dir": "./Checkpoints/",
        "test_load_weight": "ckpt_49_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledNoGuidenceImgs.png",
        "nrow": 8
        }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()

# BiO-Net
Keras implementation of "BiO-Net: Learning Recurrent Bi-directional Connections for Encoder-Decoder Architecture", MICCAI 2020

Paper: https://arxiv.org/abs/2007.00243

![BiO-Net](./images/network.png)

## Dependencies

* Python >= 3.6
* tensorflow-gpu >= 1.14.0
* PIL >= 7.0.0
* keras >= 2.1.5

## Data

**MoNuSeg**

- Baidu Disk: https://pan.baidu.com/s/1tqDzX52v8GYWXF4YfUGu1Q password: dqsr
- Google Drive: https://drive.google.com/file/d/1j7vEoq6YCBNKMoOZKPSQNciHZkVzkxGD/view?usp=sharing

**TNBC**

- Baidu Disk: https://pan.baidu.com/s/1zPWTYAEffX55c2eyb3cU0Q password: zsl1
- Google Drive: https://drive.google.com/file/d/1RYY7vE0LAHSTQXvLx41civNRZvl-2hnJ/view?usp=sharing

*NOTE:* You can place your own dataloader under ```core/dataloader.py```.

## Usage

**Train**
```
python3 train.py --epochs 300 --iter 3 --integrate --train_data PATH_TO_TRAIN_DATA_ROOT \
				 --valid_data PATH_TO_VALID_DATA_ROOT --exp 1
```

**Evaluate**
```
python3 train.py --evaluate_only --valid_data PATH_TO_VALID_DATA_ROOT --exp 1
```
or
```
python3 train.py --evaluate_only --valid_data PATH_TO_VALID_DATA_ROOT --model_path PATH_TO_TRAINED_MODEL
```

## Citation

If you find this repo useful in your work or research, please cite:

```
@inproceedings{xiang2020bionet,
  title={BiO-Net: Learning Recurrent Bi-directional Connections for Encoder-Decoder Architecture},
  author={Xiang, Tiange and Zhang, Chaoyi and Liu, Dongnan and Song, Yang and Huang, Heng and Cai, Weidong},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  month={October},
  year={2020},
}
```



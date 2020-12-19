import torch
import argparse
import os
from models import *  # set ONNX_EXPORT in models.py
import models
from utils.datasets import *
from utils.utils import *
from collections import OrderedDict

models.ONNX_EXPORT = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file')
    parser.add_argument('--inp-size', type=int, help='input size')
    parser.add_argument('--pth', type=str, help='float pth file')
    args = parser.parse_args()
    pth_name = Path(args.pth).with_suffix('').name
    
    if not os.path.isfile(args.cfg):
        raise FileNotFoundError('file not exits. %s' % args.cfg)
    if not os.path.isfile(args.pth):
        raise FileNotFoundError('file not exits. %s' % args.pth)
    
    
    model = Darknet(args.cfg, (args.inp_size, args.inp_size))
    weights = torch.load(args.pth, map_location='cpu')['model']
    for k in list(weights.keys()):
        if 'total_ops' in k or 'total_params' in k:
            weights.pop(k)
    model.load_state_dict(weights)
    darknet_w_path = str(Path(str(args.pth)).with_suffix('.weights'))
    save_weights(model, darknet_w_path)

    fake_inp = torch.randn([1, 3, args.inp_size, args.inp_size])
    onnx_file = str(Path(str(args.pth)).with_suffix('.onnx'))
    torch.onnx.export(
        model=model,
        args=fake_inp,
        f=onnx_file,
        input_names=['input'],
        output_names=[],
        opset_version=9,
    )

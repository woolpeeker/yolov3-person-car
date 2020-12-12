import torch
import argparse
import os
from models import *  # set ONNX_EXPORT in models.py
import models
from utils.datasets import *
from utils.utils import *
from collections import OrderedDict

W_FBITS = 11
B_FBITS = 10
models.ONNX_EXPORT = True

def fixpoint_quantize(x, fbits):
    x = torch.round(x * 2**fbits)
    x = x.clamp(-2**15, 2**15-1)
    x = x / 2**fbits
    return x
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file')
    parser.add_argument('--inp-size', type=int, help='input size')
    parser.add_argument('--pth', type=str, help='float pth file')
    args = parser.parse_args()
    pth_name = Path(args.pth).with_suffix('').name
    q_pth_name = Path(args.pth).parent / (pth_name + '-q.pt')
    darknet_w_path = Path(args.pth).parent / (pth_name + '.weights')
    
    if not os.path.isfile(args.cfg):
        raise FileNotFoundError('file not exits. %s' % args.cfg)
    if not os.path.isfile(args.pth):
        raise FileNotFoundError('file not exits. %s' % args.input)
    
    
    model = Darknet(args.cfg, (args.inp_size, args.inp_size))
    weights = torch.load(args.pth, map_location='cpu')['model']
    for k in list(weights.keys()):
        if 'total_ops' in k or 'total_params' in k:
            weights.pop(k)
    model.load_state_dict(weights)

    # save to darknet weights
    from models import save_weights
    save_weights(model, darknet_w_path)



    model.fuse()
    qstate_dict = OrderedDict()
    for k, v in model.state_dict().items():
        if 'weight' in k:
            v = fixpoint_quantize(v, W_FBITS)
        elif 'bias' in k:
            v = fixpoint_quantize(v, B_FBITS)
        qstate_dict[k] = v
    torch.save(qstate_dict, str(q_pth_name))
    model.load_state_dict(qstate_dict)

    fake_inp = torch.randn([1, 3, args.inp_size, args.inp_size])
    onnx_file = str(Path(str(q_pth_name)).with_suffix('.onnx'))
    torch.onnx.export(
        model=model,
        args=fake_inp,
        f=onnx_file,
        input_names=['input'],
        output_names=[],
        opset_version=9,
    )

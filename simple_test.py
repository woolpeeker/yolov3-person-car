import argparse
import json

from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.utils import *
import cv2

def preprocess(img_file, img_size):
    img = cv2.imread(str(img_file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    scale = img_size / max(img.shape[0], img.shape[1])
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    padded = np.zeros([img_size, img_size, 3], dtype=np.uint8)
    padded[:img.shape[0], :img.shape[1], :] = img
    padded = torch.tensor(padded).float().permute([2, 0, 1])
    padded = padded.unsqueeze(0)
    padded = padded.float() / 255
    return padded, scale

def preprocess2(img_file, img_size):
    img = cv2.imread(str(img_file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img = torch.tensor(img).float().permute([2, 0, 1])
    img = img.unsqueeze(0)
    img = img.float() / 255
    return padded, scale


def test(args):
    device = torch_utils.select_device(args.device, batch_size=1)
    model = Darknet(args.cfg, args.img_size)
    model.load_state_dict(torch.load(args.weights, map_location=device)['model'], strict=True)
    model.eval()
    model.to(device)
    data = parse_data_cfg(args.data)
    nc = int(data['classes'])  # number of classes
    path = Path(data['valid'])  # path to test images
    names = load_classes(data['names'])  # class names

    assert path.suffix=='.txt'
    fp = open('../mAP/input/detection-results.txt', 'w')
    for img_file in np.loadtxt(str(path), dtype=str).tolist():
        inp_tensor, scale = preprocess(img_file, args.img_size)
        inp_tensor = inp_tensor.to(device)
        with torch.no_grad():
            inf_out, train_out = model(inp_tensor, augment=False) 
            output = non_max_suppression(inf_out, conf_thres=args.conf_thres, iou_thres=0.4, multi_label=False)
        det = output[0] # batch_size = 1
        file_id = Path(img_file).with_suffix('').name
        if det is not None:
            det[:, :4] = (det[:, :4] / scale).round()
            for *xyxy, conf, cls_id in reversed(det):
                fp.write('%s %s %f %d %d %d %d\n' % (file_id, names[int(cls_id)], conf, *xyxy))
        fp.write('%s %s %f %d %d %d %d\n' % (file_id, names[0], 0, 0, 0, 1, 1))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-tiny-person-car.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/coco_person_car.data', help='*.data path')
    parser.add_argument('--weights', type=str, default='weights/v3tiny.pt', help='weights path')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1) or cpu')
    args = parser.parse_args()
    test(args)
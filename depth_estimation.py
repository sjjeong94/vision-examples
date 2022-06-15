import cv2
import torch
import videocv
import numpy as np


class Module:
    def __init__(self, model_type='DPT_Large'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net = torch.hub.load('intel-isl/MiDaS', model_type)
        net = net.eval()
        net = net.to(device)

        midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
        if model_type == 'DPT_Large' or model_type == 'DPT_Hybrid':
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform

        self.model_type = model_type
        self.device = device
        self.net = net
        self.transform = transform

        if model_type == 'DPT_Large':
            scale = 40
        elif model_type == 'DPT_Hybrid':
            scale = 3000
        else:
            scale = 500

        self.scale = scale

    @torch.inference_mode()
    def __call__(self, image):
        x = self.transform(image).to(self.device)
        prediction = self.net(x)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode='bicubic',
            align_corners=False,
        ).squeeze()
        out = prediction.cpu().numpy()
        o = (np.clip(out / self.scale, 0, 1) * 255).astype(np.uint8)
        o = cv2.applyColorMap(o, cv2.COLORMAP_MAGMA)
        return o


def main(
    video_path='./videos/test.mp4',
    save_path='./videos/depth_estimation.mp4',
    size=(640, 360),
):
    midas_small = Module('MiDaS_small')
    dpt_hybrid = Module('DPT_Hybrid')
    dpt_large = Module('DPT_Large')

    video = videocv.Video(video_path)
    writer = videocv.Writer(save_path, video.fps, (size[0]*2, size[1]*2))

    frame_count = int(video.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = 0
    while video():
        idx += 1
        print('%d / %d' % (idx, frame_count))

        image = cv2.resize(video.frame, size, interpolation=cv2.INTER_AREA)
        x = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        out_midas = midas_small(x)
        out_dpt_h = dpt_hybrid(x)
        out_dpt_l = dpt_large(x)

        color = (255, 255, 255)
        cv2.putText(out_midas, 'MiDaS_small', (10, 20), 0, 0.5, color, 1)
        cv2.putText(out_dpt_h, 'DPT_Hybrid', (10, 20), 0, 0.5, color, 1)
        cv2.putText(out_dpt_l, 'DPT_Large', (10, 20), 0, 0.5, color, 1)

        view_left = np.concatenate([image, out_midas], axis=0)
        view_right = np.concatenate([out_dpt_h, out_dpt_l], axis=0)
        view = np.concatenate([view_left, view_right], axis=1)
        writer(view)

        cv2.imshow('view', view)


if __name__ == '__main__':
    main()

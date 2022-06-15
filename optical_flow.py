import cv2
import torch
import videocv
import numpy as np
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image


def preprocess(x):
    x = x.transpose(2, 0, 1).astype(np.float32) / 255.
    x = x * 2 - 1
    x = torch.from_numpy(x).unsqueeze(0)
    return x


def visualize(flow):
    flow = flow / (np.abs(flow).max() + 1e-9) * 0.5 + 0.5
    flow_view = (flow * 255).astype(np.uint8)
    colormap = cv2.COLORMAP_TWILIGHT_SHIFTED
    flow_view = cv2.applyColorMap(flow_view, colormap)
    return flow_view


class Module:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = raft_large(pretrained=True)
        net = net.eval()
        net = net.to(device)

        self.device = device
        self.net = net

        self.prev = None
        self.next = None

    @torch.inference_mode()
    def __call__(self, image):
        x = preprocess(image).to(self.device)
        if self.prev is None:
            self.prev = x
            self.next = x
        else:
            self.prev = self.next
            self.next = x

        out = self.net(self.prev, self.next)
        flow = out[-1]
        flow_image = flow_to_image(flow)
        flow_image = flow_image.cpu().numpy().squeeze().transpose(1, 2, 0)
        flow_image = cv2.cvtColor(flow_image, cv2.COLOR_RGB2BGR)
        flow = flow.cpu().numpy().squeeze()

        return flow, flow_image


def main(
    video_path='./videos/test.mp4',
    save_path='./videos/optical_flow.mp4',
    size=(640, 360),
):
    raft = Module()

    video = videocv.Video(video_path)
    writer = videocv.Writer(save_path, video.fps, (size[0]*2, size[1]*2))

    idx = 0
    while video():
        idx += 1
        print('%d / %d' % (idx, video.frame_count))

        image = cv2.resize(video.frame, size, interpolation=cv2.INTER_AREA)
        x = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        flow, flow_image = raft(x)

        x_flow_view = visualize(flow[0])
        y_flow_view = visualize(flow[1])

        color = (0, 0, 0)
        cv2.putText(flow_image, 'Optical Flow', (10, 20), 3, 0.5, color, 1)
        cv2.putText(x_flow_view, 'X', (10, 20), 3, 0.5, color, 1)
        cv2.putText(y_flow_view, 'Y', (10, 20), 3, 0.5, color, 1)

        view_top = np.concatenate([image, flow_image], axis=1)
        view_bot = np.concatenate([x_flow_view, y_flow_view], axis=1)
        view = np.concatenate([view_top, view_bot], axis=0)
        writer(view)

        cv2.imshow('view', view)


if __name__ == '__main__':
    main()

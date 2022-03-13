import torchvision.transforms as T
import numpy as np


def make_prediction(ort_sess, img_np):
    transform = T.Compose([T.ToPILImage(), T.Resize(64),  T.ToTensor(), T.Normalize(0.5, 0.5)])
    img_np = transform(img_np).view(1, 1, 64, 64)
    outputs = ort_sess.run(None, {'input': img_np.numpy()})
    return outputs[0]

import matplotlib.pyplot as plt
import random
import torch
from fastai.vision.all import *

class ActivationHook():
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_func)   
    def hook_func(self, m, i, o): self.stored = o.detach().clone()
    def __enter__(self, *args): return self
    def __exit__(self, *args): self.hook.remove()
              
class GradientHook():
    def __init__(self, m):
        self.hook = m.register_backward_hook(self.hook_func)   
    def hook_func(self, m, gi, go): self.stored = go[0].detach().clone()
    def __enter__(self, *args): return self
    def __exit__(self, *args): self.hook.remove()
        
def calc_gradcam(label_id_pred, learner, img):
    x, = first(learner.dls.test_dl([img]))
    x_dec = TensorImage(learner.dls.train.decode((x,))[0][0])
    with GradientHook(learner.model[0]) as hookg:
        with ActivationHook(learner.model[0]) as hook:
            output = learner.model.eval()(x.cuda())
            act = hook.stored
        output[0,label_id_pred].backward()
        grad = hookg.stored

    w = grad[0].mean(dim=[1,2], keepdim=True)
    cam_map = (w * act[0]).sum(0)
    return cam_map

def plot_gradcam(learner):
    plt.figure(figsize=(50,50))
    row, col = 4, 4
    random_start = random.randint(0,len(learner.dls.valid_ds) - row * col)
    for i, item in enumerate(learner.dls.valid_ds[random_start:]):
        img, label = item
        label = learner.dls.vocab[label]
        ax = plt.subplot(row, col, i*2+1)
        ax.set_title(f'Input:{label}', fontsize=25)
        plt.imshow(img)
        with torch.no_grad():
            label_pred, label_id_pred, probs = learner.predict(img)
        ax = plt.subplot(row, col, i*2+2)
        color = "green" if label_pred==label else "red"
        prob = round(probs[label_id_pred].item(), 3)
        ax.set_title(f'GradCam:{label_pred}(Prediction probability {prob})', fontsize=25, color=color)
        cam_map = calc_gradcam (label_id_pred, learner, img)
        x, = first(learner.dls.test_dl([img]))
        x_dec = TensorImage(learner.dls.train.decode((x,))[0][0])
        x_dec.show(ctx=ax)
        ax.imshow(cam_map.detach().cpu(), alpha=0.6, extent=(0,224,224,0),
                      interpolation='bilinear', cmap='magma');
        if i == (row*col)/2 - 1:
            plt.show()
            break
import os
import PIL
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import make_grid, save_image
from collections import OrderedDict
from utils import visualize_cam, Normalize
from gradcam import GradCAM, GradCAMpp


img_dir = 'examples'
# img_name = 'collies.JPG'
# img_name = 'multiple_dogs.jpg'
# img_name = 'snake.JPEG'
img_name = "real_336_000034.jpg"
img_path = os.path.join(img_dir, img_name)
pil_img = PIL.Image.open(img_path)

normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
torch_img = torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
torch_img = F.upsample(torch_img, size=(224, 224), mode='bilinear', align_corners=False)
normed_torch_img = normalizer(torch_img)

###
# resnet = models.resnet101(pretrained=True)
resnet = models.resnet101()  #别忘记传递必要的参数
resnet101_dict = resnet.state_dict()

state_dict = torch.load('./model/2real/extractor_8.pth')	#加载预先训练好net-a的.pth文件

new_state_dict = OrderedDict()		#不是必要的【from collections import OrderedDict】

new_state_dict = {k:v for k,v in state_dict.items() if k in resnet101_dict}	#删除net-b不需要的键
resnet101_dict.update(new_state_dict)	#更新参数
resnet.load_state_dict(resnet101_dict)	#加载参数
resnet.eval(), resnet.cuda();
###


cam_dict = dict()

resnet_model_dict = dict(type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
resnet_gradcam = GradCAM(resnet_model_dict, True)
resnet_gradcampp = GradCAMpp(resnet_model_dict, True)
cam_dict['resnet'] = [resnet_gradcam, resnet_gradcampp]

images = []
for gradcam, gradcam_pp in cam_dict.values():
    mask, _ = gradcam(normed_torch_img)
    heatmap, result = visualize_cam(mask.cpu(), torch_img.cpu())

    mask_pp, _ = gradcam_pp(normed_torch_img)
    heatmap_pp, result_pp = visualize_cam(mask_pp.cpu(), torch_img.cpu())

    images.append(torch.stack([torch_img.squeeze().cpu(), heatmap, heatmap_pp, result, result_pp], 0))


# images = make_grid(torch.cat(images, 0), nrow=5)


output_dir = 'outputs'


os.makedirs(output_dir, exist_ok=True)
output_name = img_name
output_path = os.path.join(output_dir, output_name)
print(output_path)
print(images[0][0, :, :, :].size())
print(type(images[0]))

save_image(images[0][3, :, :, :], output_path)
PIL.Image.open(output_path)
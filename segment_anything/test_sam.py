
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry_feature, SamAutomaticMaskGeneratorMaskFeature
import numpy as np
import matplotlib.pyplot as plt

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    count_mask_pixel = np.zeros((sorted_anns[0]["segmentation"].shape[0], sorted_anns[0]["segmentation"].shape[1]))
    for ann in sorted_anns:
        m = ann['segmentation']
        count_mask_pixel += m
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))
    print(f"count for each pixel: {count_mask_pixel}")
    print(f"max count {count_mask_pixel.max()}, min count {count_mask_pixel.min()}")

    
def img_show(image, masks):
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.savefig("./000000000872_mask.png")
    plt.show()


imgFile="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/job/000000000872.jpg"
img=cv2.imread(imgFile)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

model_type = "vit_b"
ckpt="pretrain_weights/sam_vit_b_01ec64.pth"

sam = sam_model_registry_feature[model_type](checkpoint=ckpt)
sam = sam.to("cuda:0")
# mask_generator = SamAutomaticMaskGenerator(sam, points_per_side = 16)
mask_generator_feature = SamAutomaticMaskGeneratorMaskFeature(sam, points_per_side = 16)

# masks = mask_generator.generate(img)
# import pdb; pdb.set_trace()
# "masks", "iou_preds", "points" 
masks = mask_generator_feature.generate(img)
# import pdb; pdb.set_trace()

# img_show(img, masks)

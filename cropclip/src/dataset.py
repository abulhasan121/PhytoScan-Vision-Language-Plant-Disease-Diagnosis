"""
src/dataset.py
Dataset classes for LeafNet and PlantDoc.
"""

import os
from PIL import Image
from torch.utils.data import Dataset


class LeafNetDataset(Dataset):
    """LeafNet HuggingFace dataset wrapper."""
    def __init__(self, hf_dataset, transform, caption_to_idx):
        self.data           = hf_dataset
        self.transform      = transform
        self.caption_to_idx = caption_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img  = self.transform(item['image'].convert('RGB'))
        lbl  = self.caption_to_idx[item['caption']]
        return img, lbl


class PlantDocDataset(Dataset):
    """
    PlantDoc real-world field image dataset.
    Used ONLY as held-out test set — no training on PlantDoc.
    """
    def __init__(self, root: str, transform):
        self.transform = transform
        self.samples   = []
        for cls in sorted(os.listdir(root)):
            folder = os.path.join(root, cls)
            if not os.path.isdir(folder): continue
            for f in os.listdir(folder):
                if f.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
                    self.samples.append((os.path.join(folder, f), cls))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls = self.samples[idx]
        img       = self.transform(Image.open(path).convert('RGB'))
        return img, cls


# PlantDoc folder → LeafNet caption mapping
CLASS_MAPPING = {
    'Apple Scab Leaf'            : 'a image of Apple leaves diseased by Scab with symptoms of olive-green to dark brown spots on leaves, leading to yellowing',
    'Apple leaf'                 : 'a image of Apple healthy leaves with leaves appearing normal and healthy',
    'Apple rust leaf'            : 'a image of Apple leaves diseased by Cedar-Apple Rust with symptoms of yellow to bright orange-red spots, black dots, and finger-like fungal tubes on the leaf surface.',
    'Bell_pepper leaf'           : 'a image of Pepper healthy leaves with leaves appearing normal and healthy',
    'Bell_pepper leaf spot'      : 'a image of Pepper leaves diseased by Bell pepper leaf spot with symptoms of water-soaked lesions,dark purple-brown spots with light brown centers',
    'Blueberry leaf'             : 'a image of Strawberry healthy leaves with leaves appearing normal and healthy',
    'Cherry leaf'                : 'a image of Cherry healthy leaves with leaves appearing normal and healthy',
    'Corn Gray leaf spot'        : 'a image of Maize leaves diseased by Cercospora leaf spot Gray leaf spot with symptoms of small necrotic spots with halos expanding into rectangular gray to brown lesions, typically 1/8 inch wide and 2 to 3 inches long, with distinct parallel edges limited by veins',
    'Corn leaf blight'           : 'a image of Maize leaves diseased by Northern Leaf Blight with symptoms of canoe-shaped lesions with gray-green margins that turn tan with dark fungal sporulation, starting on lower leaves and spreading upwards',
    'Corn rust leaf'             : 'a image of Maize leaves diseased by Rust with symptoms of rust-colored to dark brown elongated pustules on both leaf surfaces, containing cinnamon brown urediniospores, with leaves potentially showing chlorosis and death under severe infection.',
    'Peach leaf'                 : 'a image of Peach healthy leaves with leaves appearing normal and healthy',
    'Potato leaf'                : 'a image of Potato healthy leaves with leaves appearing normal and healthy',
    'Potato leaf early blight'   : 'a image of Potato leaves diseased by Early blight with symptoms of small, dark, papery flecks growing into brown-black, circular-to-oval areas with angular borders',
    'Potato leaf late blight'    : 'a image of Potato leaves diseased by Late blight with symptoms of small, irregular green spots with pale green to yellow rings, growing across veins',
    'Raspberry leaf'             : 'a image of Raspberry healthy leaves with leaves appearing normal and healthy',
    'Soyabean leaf'              : 'a image of Soybean healthy leaves with leaves appearing normal and healthy',
    'Squash Powdery mildew leaf' : 'a image of Squash leaves diseased by Powdery mildew with symptoms of white or grayish powdery mold on leaves',
    'Strawberry leaf'            : 'a image of Strawberry healthy leaves with leaves appearing normal and healthy',
    'Tomato Early blight leaf'   : 'a image of Tomato leaves diseased by Early blight with symptoms of small, dark, papery flecks growing into brown-black, circular-to-oval areas with angular borders',
    'Tomato Septoria leaf spot'  : 'a image of Tomato leaves diseased by Septoria Leaf Spot with symptoms of numerous small, circular spots of 3-5 mm in diameter with dark-brown margins and tan to gray centers',
    'Tomato leaf'                : 'a image of Tomato healthy leaves with leaves appearing normal and healthy',
    'Tomato leaf bacterial spot' : 'a image of Tomato leaves diseased by Bacterial spot with symptoms of small, water-soaked spots that enlarge and turn dark brown',
    'Tomato leaf disease (early_blight)' : 'a image of Tomato leaves diseased by Early blight with symptoms of small, dark, papery flecks growing into brown-black, circular-to-oval areas with angular borders',
    'Tomato leaf late blight'    : 'a image of Tomato leaves diseased by Late blight with symptoms of small, irregular green spots with pale green to yellow rings, growing across veins',
    'Tomato leaf mosaic virus'   : 'a image of Tomato leaves diseased by Mosaic virus with symptoms of a mosaic pattern of light green, yellow, and dark green on leaves',
    'Tomato leaf yellow virus'   : 'a image of Tomato leaves diseased by Yellow leaf curl virus with symptoms of upward curling of leaves with yellowing edges',
    'Tomato mold leaf'           : 'a image of Tomato leaves diseased by Leaf Mold with symptoms of pale greenish-yellow spots on the upper side of leaves',
    'Tomato two spotted spider mites leaf' : 'a image of Tomato leaves diseased by Two-spotted spider mite with symptoms of tiny yellow dots on leaves with bronze discoloration',
    'grape leaf'                 : 'a image of Grape healthy leaves with leaves appearing normal and healthy',
    'grape leaf black rot'       : 'a image of Grape leaves diseased by Black Rot with symptoms of tan to brown circular spots on leaves with black borders',
}

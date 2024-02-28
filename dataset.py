from torch.utils.data import Dataset
from PIL import Image
import os
from tools.utils import get_landmark, transforms


front_landmark_path = './tools/mask120.png'


class ImageData(Dataset):
    def __init__(self, profile_dir, front_dir, transform=None):
        super(ImageData, self).__init__()
        self.profile_dir = profile_dir
        self.front_dir = front_dir
        self.profile_face_list = os.listdir(profile_dir)
        self.frontal_face_list = os.listdir(front_dir)
        self.transform = transform
        self.trans_to_60 = transforms.Compose([
            transforms.CenterCrop((120, 120)),
            transforms.Resize((60, 60)),
            transforms.ToTensor()
        ])
        self.trans_to_30 = transforms.Compose([
            transforms.CenterCrop((120, 120)),
            transforms.Resize((30, 30)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.profile_face_list)

    def __getitem__(self, idx):
        data = {}
        img_name = self.profile_face_list[idx]
        token = img_name.split('_')
        front_img_name = '_'.join(token[:3]) + '_051_' + token[4] + '_crop_128.png'
        profile_img_path = os.path.join(self.profile_dir, img_name)
        front_img_path = os.path.join(self.front_dir, front_img_name)
        with Image.open(profile_img_path) as profile_face:
            if self.transform is not None:
                data['profile'] = self.transform(profile_face)
            else:
                raise NameError('The transform is None')
        with Image.open(front_img_path) as front_face:
            if self.transform is not None:
                data['front'] = self.transform(front_face)
            else:
                raise NameError('The transform is None')
        data['real_lm'] = get_landmark(profile_img_path)
        data['target_lm'] = get_landmark(front_landmark_path, 'front')
        data['img_60'] = self.trans_to_60(front_face)
        data['img_30'] = self.trans_to_30(front_face)

        return data

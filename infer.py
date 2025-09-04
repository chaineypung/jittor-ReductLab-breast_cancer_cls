import jittor as jt
import jittor.nn as nn
from jittor.dataset import Dataset
from jittor.transform import Compose, Resize, CenterCrop, ToTensor, ImageNormalize
import glob
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import argparse
import albumentations as A
import pandas as pd
import random
import math
import sys

sys.path.append("./models")
from jimm import seresnext101_32x8d, seresnext50_32x4d, tf_efficientnetv2_s_in21k
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
import random
import cv2
from convnextv1 import Convnext_small
from collections import defaultdict, Counter


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)
    jt.flags.use_cuda = 1
    os.environ['PYTHONHASHSEED'] = str(seed)


class ImageFolder(Dataset):
    def __init__(self, root, df=None, transform=None, mode="train", **kwargs):
        super().__init__(**kwargs)
        self.root = root
        self.transform = transform
        if df is not None:
            data_dir = df[["filename", "class"]].values.tolist()
        else:
            data_dir = sorted(os.listdir(root))
            data_dir = [(x, None) for x in data_dir]
        self.data_dir = data_dir
        self.total_len = len(self.data_dir)
        self.mode = mode
        self.num_classes = 6

    def __getitem__(self, idx):

        image_path, label = os.path.join(self.root, self.data_dir[idx][0]), self.data_dir[idx][1]
        image = Image.open(image_path).convert('L')
        image = image.convert("RGB")
        if self.transform:
            if self.mode == "train":
                transform_train1, data_transforms = self.transform
                image = np.array(image)
                image = data_transforms["train"](image=image)["image"]
                image = Image.fromarray(image) 
                image = transform_train1(image)
            elif self.mode == "val":
                image = self.transform(image)
            else:
                transform_test, data_transforms = self.transform
                image = np.array(image)
                image = data_transforms["val"](image=image)["image"]
                image = Image.fromarray(image)  
                image = transform_test(image)

        image_name = self.data_dir[idx][0]
        label = image_name

        return jt.array(image), label


def l2_norm(input, axis=1):
    norm = jt.norm(input, 2, axis, True)
    output = jt.divide(input, norm)
    return output


def euclidean_dist(x, y):
    m, n = x.shape[0], y.shape[0]
    xx = (x ** 2).sum(dim=1, keepdims=True).expand(m, n)
    yy = (y ** 2).sum(dim=1, keepdims=True).expand(n, m).transpose(0, 1)
    dist = xx + yy - 2 * jt.matmul(x, y.transpose(0, 1))
    dist = jt.clamp(dist, min_v=1e-12).sqrt()  
    return dist


def hard_example_mining(dist_mat, is_pos):
    dist_ap = dist_mat[is_pos].max()
    dist_an = dist_mat[is_pos == False].min()
    return dist_ap, dist_an


class MagrginLinear(nn.Module):
   
    def __init__(self, embedding_size=512, classnum=6, s=64., m=0.5):
        super(MagrginLinear, self).__init__()
        self.classnum = classnum

        self.kernel = nn.Parameter(jt.random((embedding_size, classnum), dtype=jt.float32) * 2 - 1)
        norm = self.kernel.norm(p=2, dim=1, keepdim=True)
        norm = jt.maximum(norm, jt.float32(1e-5))
        self.kernel = self.kernel / norm * 1e5

        self.m = m 
        self.s = s 
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  
        self.threshold = math.cos(math.pi - m)

    def execute(self, embbedings, label, is_infer=False):
       
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = jt.matmul(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1) 
        cos_theta_2 = jt.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = jt.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)

        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm) 
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0  
        idx_ = jt.arange(0, nB).int64()

        if not is_infer:
            indices = jt.argmax(label, dim=1)[0]
            output[idx_, indices] = cos_theta_m[idx_, indices]

        output *= self.s 
        return output


class MarginHead(nn.Module):

    def __init__(self, num_class=6, emb_size=2048, s=64., m=0.5):
        super(MarginHead, self).__init__()
        self.fc = MagrginLinear(embedding_size=emb_size, classnum=num_class, s=s, m=m)

    def execute(self, fea, label, is_infer):
        fea = l2_norm(fea)
        logit = self.fc(fea, label, is_infer)
        return logit


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = jt.ones(1) * p
        self.eps = eps

    def execute(self, x):
        bs, ch, h, w = x.shape
        x = jt.clamp(x, min_v=self.eps).pow(self.p)
        pool = nn.AvgPool2d(kernel_size=(h, w))
        x = pool(x)
        x = x.pow(1.0 / self.p)
        x = x.reshape(bs, ch)
        return x


class Net_effv2s(nn.Module):
    def __init__(self, num_classes, pretrain):
        super().__init__()
        self.base_net = tf_efficientnetv2_s_in21k(num_classes=num_classes, pretrained=pretrain)

        self.s1 = 64
        self.m1 = 0.5
        self.s2 = 64
        emb_size = 1280
        self.fea_bn = nn.BatchNorm1d(emb_size)
        self.fea_bn.bias.stop_grad()
        self.margin_head = MarginHead(num_classes, emb_size=emb_size, s=self.s1, m=self.m1)

    def execute(self, x, label=None, is_infer=False):
        x = self.base_net.conv_stem(x)
        x = self.base_net.bn1(x)
        x = self.base_net.act1(x)
        x = self.base_net.blocks(x)
        x = self.base_net.conv_head(x)
        x = self.base_net.bn2(x)
        x = self.base_net.act2(x)

        x = self.base_net.global_pool(x)
        fea = x
        fea = self.fea_bn(fea)

        logit = self.base_net.classifier(l2_norm(fea)) * 16.0
        logit_margin = self.margin_head(fea, label=label, is_infer=is_infer)

        return logit, logit_margin, fea


class Net_convnextv1s(nn.Module):
    def __init__(self, num_classes, pretrain):
        super().__init__()
        self.base_net = Convnext_small(pretrained=pretrain)

        self.s1 = 64
        self.m1 = 0.5
        self.s2 = 64
        emb_size = 768
        self.fea_bn = nn.BatchNorm1d(emb_size)
        self.fea_bn.bias.stop_grad()

        self.gem = GeM(p=3, eps=1e-6)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(emb_size, num_classes)
        self.margin_head = MarginHead(num_classes, emb_size=emb_size, s=self.s1, m=self.m1)

    def execute(self, x, label=None, is_infer=False):
        feats = self.base_net(x)
        x = feats[-1]

        fea = self.pool(x)
        fea = fea.squeeze(-1).squeeze(-1)

        fea = self.fea_bn(fea)
        logit = self.fc(l2_norm(fea)) * 16.0
        logit_margin = self.margin_head(fea, label=label, is_infer=is_infer)

        return logit, logit_margin, fea


class Net_seresnext101(nn.Module):
    def __init__(self, num_classes, pretrain):
        super().__init__()
        self.base_net = seresnext101_32x8d(num_classes=num_classes, pretrained=pretrain)

        self.s1 = 64
        self.m1 = 0.5
        self.s2 = 64
        emb_size = 2048
        self.fea_bn = nn.BatchNorm1d(emb_size)
        self.fea_bn.bias.stop_grad()
        self.margin_head = MarginHead(num_classes, emb_size=emb_size, s=self.s1, m=self.m1)

    def execute(self, x, label=None, is_infer=False):
        x = self.base_net.conv1(x)
        x = self.base_net.bn1(x)
        x = self.base_net.act1(x)
        x = self.base_net.maxpool(x)

        x = self.base_net.layer1(x)
        x = self.base_net.layer2(x)
        x = self.base_net.layer3(x)
        x = self.base_net.layer4(x)

        x = self.base_net.global_pool(x)

        fea = x
        fea = self.fea_bn(fea)
        logit = self.base_net.fc(l2_norm(fea)) * 16.0
        logit_margin = self.margin_head(fea, label=label, is_infer=is_infer)

        return logit, logit_margin, fea


def evaluate(model: nn.Module, val_loader: Dataset):
    model.eval()
    sigmoid = nn.Sigmoid()
    preds, targets = [], []
    probs = []
    print("Evaluating...")
    for data in val_loader:
        image, label = data
        pred, _, _ = model(image, None, is_infer=True)
        pred.sync()

        prob = sigmoid(pred).numpy()
        true_labels = np.argmax(label.numpy(), axis=1)

        probs.append(prob)
        preds.append(prob.argmax(axis=1))
        targets.append(true_labels)

    probs = np.concatenate(probs)
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    acc = accuracy_score(targets, preds)
    balanced_acc = balanced_accuracy_score(targets, preds)

    try:
        y_true = np.eye(6)[targets] 
        auc = roc_auc_score(y_true, probs, average='macro')
    except:
        auc = float('nan')

    return acc, auc, balanced_acc




# ============== Test ==================
# 原始
def test1(models, test_loader: Dataset, result_path: str):
    for model in models:
        model.eval()

    preds = []
    names = []
    print("Testing...")
    for data in test_loader:
        image, image_names = data
        ensemble_prob = None

        for model in models:
            with jt.no_grad():
                pred, _, _ = model(image, None, is_infer=True)  # .numpy()
                pred = pred.numpy()
                # pred = nn.softmax(pred, dim=1).numpy()
                if ensemble_prob is None:
                    ensemble_prob = pred
                else:
                    ensemble_prob += pred

        ensemble_prob /= len(models)
        pred = ensemble_prob.argmax(axis=1)

        preds.append(pred)
        names.extend(image_names)

    preds = np.concatenate(preds)
    with open(result_path, 'w') as f:
        for name, pred in zip(names, preds):
            f.write(f'{name} {pred}\n')


# 模型变换均取众数
def test2(models, test_loader, result_path: str):
    for model in models:
        model.eval()

    print("Testing...")

    all_votes = defaultdict(list)

    for data in test_loader:
        images, image_names = data  # images: (B, C, H, W)

        augmented_batches = apply_tta_batch(images)

        for aug_images in augmented_batches:
            for model in models:
                with jt.no_grad():
                    preds, _, _ = model(aug_images, None, is_infer=True)
                    preds = preds.numpy()
                    pred_classes = preds.argmax(axis=1)

                for name, pred in zip(image_names, pred_classes):
                    all_votes[name].append(int(pred))

    final_results = []
    for name, votes in all_votes.items():
        counts = np.bincount(votes)
        majority_class = np.argmax(counts)
        final_results.append((name, majority_class))

    with open(result_path, 'w') as f:
        for name, pred in final_results:
            f.write(f"{name} {pred}\n")


# 模型变换均取平均
def test3(models, test_loader, result_path: str):
    for model in models:
        model.eval()

    print("Testing...")

    all_probs = defaultdict(list)

    for data in test_loader:
        images, image_names = data

        augmented_batches = apply_tta_batch(images)

        for aug_images in augmented_batches:
            for model in models:
                with jt.no_grad():
                    preds, _, _ = model(aug_images, None, is_infer=True)
                    one_hot = preds.numpy()

                for name, prob in zip(image_names, one_hot):
                    all_probs[name].append(prob)

    final_results = []
    for name, prob_list in all_probs.items():
        avg_prob = np.mean(prob_list, axis=0)
        pred_class = int(np.argmax(avg_prob))
        final_results.append((name, pred_class))

    with open(result_path, 'w') as f:
        for name, pred in final_results:
            f.write(f"{name} {pred}\n")


# 模型之间平均变换之间众数
def test4(models, test_loader, result_path: str):
    for model in models:
        model.eval()

    print("Testing...")
    all_votes = defaultdict(list)

    for data in test_loader:
        images, image_names = data

        tta_preds_per_image = [[] for _ in range(len(image_names))]
        augmented_batches = apply_tta_batch(images)

        for aug_images in augmented_batches:
            ensemble_probs = None

            for model in models:
                with jt.no_grad():
                    preds, _, _ = model(aug_images, None, is_infer=True)
                    preds = preds.numpy()

                    if ensemble_probs is None:
                        ensemble_probs = preds
                    else:
                        ensemble_probs += preds

            ensemble_probs /= len(models)
            pred_classes = ensemble_probs.argmax(axis=1)

            for idx, pred in enumerate(pred_classes):
                tta_preds_per_image[idx].append(int(pred))

        for name, votes in zip(image_names, tta_preds_per_image):
            counts = np.bincount(votes)
            majority_class = np.argmax(counts)
            all_votes[name] = majority_class

    with open(result_path, 'w') as f:
        for name in sorted(all_votes.keys()):
            f.write(f"{name} {all_votes[name]}\n")


# 模型之间众数变换之间平均
def to_one_hot(indices, num_classes):
    return np.eye(num_classes)[indices]
def test5(models, test_loader, result_path: str, num_classes=10):
    for model in models:
        model.eval()

    print("Testing...")
    final_results = {}

    for data in test_loader:
        images, image_names = data
        batch_size = len(image_names)
        augmented_batches = apply_tta_batch(images)
        model_votes_per_image = [[] for _ in range(batch_size)]

        for model in models:

            tta_preds_per_image = [[] for _ in range(batch_size)]
            for aug_images in augmented_batches:
                with jt.no_grad():
                    preds, _, _ = model(aug_images, None, is_infer=True)
                    preds = preds.numpy()
                    pred_classes = preds.argmax(axis=1)

                    for idx, pred in enumerate(pred_classes):
                        tta_preds_per_image[idx].append(int(pred))

            for idx in range(batch_size):
                votes = tta_preds_per_image[idx]
                most_common = Counter(votes).most_common(1)[0][0]
                model_votes_per_image[idx].append(most_common)

        for idx, name in enumerate(image_names):
            model_preds = model_votes_per_image[idx]
            one_hot_preds = to_one_hot(model_preds, num_classes)
            avg_prediction = one_hot_preds.mean(axis=0)
            final_class = avg_prediction.argmax()
            final_results[name] = int(final_class)

    with open(result_path, 'w') as f:
        for name in sorted(final_results.keys()):
            f.write(f"{name} {final_results[name]}\n")


# ============== Main ==============
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='./TestSetB')
    parser.add_argument('--testonly', action='store_true', default=True)
    parser.add_argument('--result_path', type=str, default='./result.txt')
    args = parser.parse_args()

    set_seed(42)

    data_transforms = {
        "train": A.Compose([
            A.Resize(height=512, width=512, p=1.0),
            A.RandomCrop(height=448, width=448, p=1.0),

            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Downscale(p=0.25),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=60, p=0.5),

            A.OneOf([
                A.RandomToneCurve(scale=0.15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.3, 0.3), p=0.5)
            ], p=0.5),

            A.CoarseDropout(max_holes=6, max_height=0.15, max_width=0.25, min_holes=1, min_height=0.05, min_width=0.1,
                            fill_value=0, mask_fill_value=None, p=0.25),

        ], p=1.0),
        "val": A.Compose([
            A.Resize(height=512, width=512, p=1.0),
            A.RandomCrop(height=448, width=448, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Downscale(p=0.25),
            A.ShiftScaleRotate(shift_limit=0.1,
                               scale_limit=0.15,
                               rotate_limit=60,
                               p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=0.2,
                sat_shift_limit=0.2,
                val_shift_limit=0.2,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=(-0.1, 0.1),
                p=0.5
            ),
        ], p=1.)
    }
    transform_train1 = Compose([
        ToTensor(),
        ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_val = Compose([
        Resize((512, 512)),
        CenterCrop(448+32),
        ToTensor(),
        ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_test = Compose([
        ToTensor(),
        ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


    test_loader = ImageFolder(
        root=args.dataroot,
        transform=transform_val,
        batch_size=8,
        num_workers=8,
        shuffle=False,
        mode="val"
    )

    # ===================================================================================

    def apply_tta_batch(batch_images):
        def rot90(x, k):
            if k == 1:
                x = x.transpose(2, 3)
                return jt.flip(x, dim=[2])
            elif k == 2:
                return jt.flip(x, dim=[2, 3])
            elif k == 3:
                x = x.transpose(2, 3)
                return jt.flip(x, dim=[3])
            return x

        return [
            batch_images,  # 原图
            # rot90(batch_images, 1),  # 旋转 90°
            # rot90(batch_images, 2),  # 旋转 180°
            # rot90(batch_images, 3),  # 旋转 270°
            # jt.flip(batch_images, dim=[2]),  # 垂直翻转
            # jt.flip(batch_images, dim=[3]),  # 水平翻转
        ]

    models = []
    model_weight_path = "../checkpoint/checkpoint1.pkl"
    model_weight = jt.load(model_weight_path)

    MODE1 = ["EMA"]
    MODE2 = ["LB"]
    FOLD = [...]
    for mode1 in MODE1:
        for mode2 in MODE2:
            for fold in FOLD:
                model = Net_convnextv1s(pretrain=False, num_classes=6)
                model.load_parameters(model_weight[mode1][mode2][fold])
                models.append(model)

    model_weight_path = "../checkpoint/checkpoint2.pkl"
    model_weight = jt.load(model_weight_path)

    MODE1 = ["EMA"]
    MODE2 = ["LB"]
    FOLD = [...]
    for mode1 in MODE1:
        for mode2 in MODE2:
            for fold in FOLD:
                model = Net_effv2s(pretrain=False, num_classes=6)
                model.load_parameters(model_weight[mode1][mode2][fold])
                models.append(model)

    test2(models, test_loader, args.result_path)

    # ===================================================================================

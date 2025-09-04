import jittor as jt
import jittor.nn as nn
from jittor.dataset import Dataset
from jittor.transform import Compose, Resize, CenterCrop, ToTensor, ImageNormalize
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import argparse
import albumentations as A
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
import random
import math
import sys
sys.path.append("/defaultShare/archive/pengchenxu/breast_cancer/models")
from jimm import seresnext101_32x8d, seresnext50_32x4d
import random
import cv2
from convnextv1 import Convnext_small


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)  
    jt.flags.use_cuda = 1
    os.environ['PYTHONHASHSEED'] = str(seed)

    
def generate_smooth_label(label, num_classes, smooth_adjacent=0.1, smooth_far=0.05):
    target = np.full(num_classes, smooth_far)
    target[label] = 0.0
    if label - 1 >= 0:
        target[label - 1] = smooth_adjacent
    if label + 1 < num_classes:
        target[label + 1] = smooth_adjacent
    # target[label] = 0.9
    target[label] = 1.0 - target.sum()
    return target


def label_smooth(label, num_classes, smooth=0.1):
    one_hot = np.zeros(num_classes)
    one_hot[label] = 1
    smooth_val = smooth / (num_classes - 1)
    smoothed = one_hot * (1.0 - smooth) + (1.0 - one_hot) * smooth_val
    return smoothed


# ============== Dataset ==============
class ImageFolder(Dataset):
    def __init__(self, root, df=None, transform=None, mode="train",**kwargs):
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
                transform_train1,data_transforms = self.transform
                image = np.array(image) 
                image = data_transforms["train"](image=image)["image"]
                image = Image.fromarray(image)  #
                image = transform_train1(image)
            elif self.mode == "val":
                image = self.transform(image)
            else:
                transform_test,data_transforms = self.transform
                image = np.array(image)  
                image = data_transforms["val"](image=image)["image"]
                image = Image.fromarray(image)  #
                image = transform_test(image)

        image_name = self.data_dir[idx][0]
        # if label is not None:
        #     one_hot_label = np.zeros(self.num_classes)
        #     one_hot_label[label] = 1
        #     return jt.array(image), jt.array(one_hot_label)
        # else:
        #     label = image_name
        # return jt.array(image), label
        
        if label is not None:
            if self.mode == "train":
                one_hot_label = generate_smooth_label(label, self.num_classes, smooth_adjacent=0.1, smooth_far=0.05)
                # one_hot_label = label_smooth(label, self.num_classes, smooth=0.1)
                return jt.array(image), jt.array(one_hot_label)
            else:
                one_hot_label = np.zeros(self.num_classes)
                one_hot_label[label] = 1
                return jt.array(image), jt.array(one_hot_label)
        else:
            label = image_name

        return jt.array(image), label
    
    
# ============== Loss ==============
class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def execute(self, preds, targets):
        bce_loss = nn.BCEWithLogitsLoss(size_average=False)(preds, targets)
        probas = nn.Sigmoid()(preds)
        loss = targets * self.alpha * \
               (1. - probas) ** self.gamma * bce_loss + \
               (1. - targets) * probas ** self.gamma * bce_loss
        loss = loss.mean()
        return loss
    
def l2_norm(input, axis=1):
    norm = jt.norm(input,2, axis, True)
    output = jt.divide(input, norm)
    return output

def softmax_loss(results, labels, one_hot=True):
    if one_hot:
        labels = jt.argmax(labels, dim=1)[0]
    loss = nn.cross_entropy_loss(results, labels)
    return loss

def euclidean_dist(x, y):
    m, n = x.shape[0], y.shape[0]
    xx = (x ** 2).sum(dim=1, keepdims=True).expand(m, n)
    yy = (y ** 2).sum(dim=1, keepdims=True).expand(n, m).transpose(0, 1)
    dist = xx + yy - 2 * jt.matmul(x, y.transpose(0, 1))
    dist = jt.clamp(dist, min_v=1e-12).sqrt()  # clamp 改为 min_v
    return dist

def hard_example_mining(dist_mat, is_pos):
    # hardest positive: max over positive distances
    dist_ap = dist_mat[is_pos].max()
    
    # hardest negative: min over negative distances
    dist_an = dist_mat[is_pos == False].min()

    return dist_ap, dist_an

class TripletLoss(object):

    def __init__(self, margin=None):
        self.margin = margin

    def __call__(self, global_feat, labels, one_hot=True):
        global_feat = l2_norm(global_feat)

        dist_mat = euclidean_dist(global_feat, global_feat)
        if one_hot:
            labels = jt.argmax(labels, dim=1)[0]
        
        NB = len(labels)
        labels = labels.reshape(-1, 1)
        is_pos_all = (labels == labels.transpose())
        losses = []
        for i in range(NB):
            is_pos = is_pos_all[i]
            dist_ = dist_mat[i]        
            dist_ap, dist_an = hard_example_mining(dist_, is_pos)
            if self.margin is not None:
                loss = jt.maximum(0, -(dist_an - dist_ap) + self.margin)
            else:
                loss = jt.log(1 + jt.exp(-(dist_an - dist_ap)))
            losses.append(loss)

        return sum(losses) / NB
    
# ============== Model ==============
class MagrginLinear(nn.Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=512, classnum=6,  s=64., m=0.5):
        super(MagrginLinear, self).__init__()
        self.classnum = classnum

        self.kernel = nn.Parameter(jt.random((embedding_size, classnum), dtype=jt.float32) * 2 - 1)
        # initial kernel
        norm = self.kernel.norm(p=2, dim=1, keepdim=True)
        norm = jt.maximum(norm, jt.float32(1e-5))
        self.kernel = self.kernel / norm * 1e5

        self.m = m # the margin value, default is 0.5
        self.s = s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)

    def execute(self, embbedings, label, is_infer = False):
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel, axis=0)
        # cos(theta+m)
        cos_theta = jt.matmul(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        cos_theta_2 = jt.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = jt.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)

        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm) # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0 # a little bit hacky way to prevent in_place operation on cos_theta
        idx_ = jt.arange(0, nB).int64()

        if not is_infer:
            indices = jt.argmax(label, dim=1)[0]
            output[idx_, indices] = cos_theta_m[idx_, indices]

        output *= self.s # scale up in order to make softmax work, first introduced in normface
        return output

class MarginHead(nn.Module):

    def __init__(self, num_class=6, emb_size = 2048, s=64., m=0.5):
        super(MarginHead,self).__init__()
        self.fc = MagrginLinear(embedding_size=emb_size, classnum=num_class , s=s, m=m)

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
    
    
# class Net(nn.Module):
#     def __init__(self, num_classes, pretrain):
#         super().__init__()
#         self.base_net = seresnext50_32x4d(num_classes=num_classes, pretrained=pretrain)

#         self.s1 = 64
#         self.m1 = 0.5
#         self.s2 = 64
#         emb_size = 2048
#         self.fea_bn = nn.BatchNorm1d(emb_size)
#         self.fea_bn.bias.stop_grad()
        
#         self.gem = GeM(p=3, eps=1e-6)

#         self.margin_head = MarginHead(num_classes, emb_size=emb_size, s = self.s1, m = self.m1)

#     def execute(self, x, label=None, is_infer=False):
#         x = self.base_net.conv1(x)
#         x = self.base_net.bn1(x)
#         x = self.base_net.act1(x)
#         x = self.base_net.maxpool(x)

#         x = self.base_net.layer1(x)
#         x = self.base_net.layer2(x)
#         x = self.base_net.layer3(x)
#         x = self.base_net.layer4(x)

#         # x = self.base_net.global_pool(x)

#         x = self.gem(x)
#         print("fea shape:", x.shape)
#         fea = x
#         fea = self.fea_bn(fea)
        
#         logit = self.base_net.fc(l2_norm(fea)) * 16.0
#         logit_margin = self.margin_head(fea, label = label, is_infer=is_infer)

#         return logit, logit_margin, fea

class Net(nn.Module):
    def __init__(self, num_classes, pretrain):
        super().__init__()
        self.base_net = Convnext_small(pretrained=pretrain)

        self.s1 = 64
        self.m1 = 0.5
        self.s2 = 64
        emb_size = 768  
        self.fea_bn = nn.BatchNorm1d(emb_size)
        self.fea_bn.bias.stop_grad()
        
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(emb_size, num_classes)  
        self.margin_head = MarginHead(num_classes, emb_size=emb_size, s=self.s1, m=self.m1)

    def execute(self, x, label=None, is_infer=False):
        feats = self.base_net(x)             # tuple, 每个stage
        x = feats[-1]                        # [B, C, H, W]

        fea = self.pool(x)        # [B, C, 1, 1]
        fea = fea.squeeze(-1).squeeze(-1)   # [B, C]
        # print("fea shape:", fea.shape)


        fea = self.fea_bn(fea)
        logit = self.fc(l2_norm(fea)) * 16.0
        logit_margin = self.margin_head(fea, label=label, is_infer=is_infer)

        return logit, logit_margin, fea
    
    
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return bbx1, bby1, bbx2, bby2, lam


def mixup(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.shape[0]
    index = np.random.permutation(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    mixed_y = lam * y_a + (1 - lam) * y_b
    return mixed_x, mixed_y


def cutmix(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.shape[0]
    index = np.random.permutation(batch_size)
    bbx1, bby1, bbx2, bby2, lam = rand_bbox(x.size(), lam)
    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    y_a, y_b = y, y[index]
    mixed_y = lam * y_a + (1 - lam) * y_b
    return mixed_x, mixed_y


# ============== Training ==============
def training(model:nn.Module, optimizer:nn.Optimizer, scheduler:nn.Optimizer, train_loader:Dataset, now_epoch:int, num_epochs:int):
    model.train()
    losses = []
    pbar = tqdm(train_loader, total=len(train_loader), bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]" + " " * (80 - 10 - 10 - 10 - 10 - 3))
    step = 0
    print(optimizer.state_dict()['defaults']['lr'])
    for data in pbar:

        step += 1
        image, label = data
        
        if random.random() < 0.0:
            # image, label = mixup(image, label, alpha=1.0)
            image, label = cutmix(image, label, alpha=1.0)
        
        pred, pred_margin, fea = model(image, label, is_infer = False)
        loss_focal = BCEFocalLoss()(pred, label)
        loss_bce = nn.BCEWithLogitsLoss(size_average=False)(pred, label)
        loss_arcface = softmax_loss(pred_margin, label)
        loss_triplet = TripletLoss(margin=0.3)(fea, label)
        loss = loss_focal * 0.1 + loss_bce + loss_arcface * 0.1 + loss_triplet
        # loss = nn.cross_entropy_loss(pred, label)
        loss.sync()
        optimizer.step(loss)
        losses.append(loss.item())
        pbar.set_description(f'Epoch {now_epoch} [TRAIN] loss = {losses[-1]:.4f}')
    
    if scheduler is not None:
        scheduler.step()

    print(f'Epoch {now_epoch} / {num_epochs} [TRAIN] mean loss = {np.mean(losses):.4f}')


def evaluate(model:nn.Module, val_loader:Dataset):
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
        
        # prob = nn.softmax(pred, dim=1).numpy()
        # probs.append(prob)
        
        # pred = prob.argmax(axis=1)
        # preds.append(pred)
        # targets.append(label.numpy())
    
    probs = np.concatenate(probs)
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    
    acc = accuracy_score(targets, preds)
    balanced_acc = balanced_accuracy_score(targets, preds)

    try:
        # 使用Sigmoid时（多标签模式）
        y_true = np.eye(6)[targets]  # 将类别索引转为one-hot
        auc = roc_auc_score(y_true, probs, average='macro')
        
        # 单标签
        # auc = roc_auc_score(targets, probs, multi_class='ovr', average='macro')
    except:
        auc = float('nan')  
    
    return acc, auc, balanced_acc


def run(model:nn.Module, optimizer:nn.Optimizer, scheduler:nn.Optimizer, train_loader:Dataset, val_loader:Dataset, num_epochs:int, modelroot:str, fold:int):
    best_balabced_acc = 0
    best_acc = 0
    best_auc = 0
    best_avg = 0
    log_path = os.path.join(modelroot, 'training_log.csv')  
    early_stop_counter = 0

    for epoch in range(num_epochs):
        training(model, optimizer, scheduler, train_loader, epoch, num_epochs)
        acc, auc, balanced_acc = evaluate(model, val_loader)
        if auc == float('nan'):
            avg_score = (acc + balanced_acc) / 2
        else:
            avg_score = (acc + auc + balanced_acc) / 3

        log_df = pd.DataFrame([{
            'fold': fold,
            'epoch': epoch,
            'acc': acc,
            'auc': auc,
            'balanced_acc': balanced_acc,
            'avg_score': avg_score,
            'best_avg_score': max(best_avg, avg_score)
        }])
        
        log_df.to_csv(log_path, mode='a', header=False, index=False)
        
        if epoch >= 15:
            model.save(os.path.join(modelroot, f'fold{fold}_epoch{epoch}.pkl'))
        
        if acc > best_acc:
            best_acc = acc
            model.save(os.path.join(modelroot, f'fold{fold}_best_acc.pkl'))
            early_stop_counter = 0
        if auc > best_auc:
            best_auc = auc
            model.save(os.path.join(modelroot, f'fold{fold}_best_auc.pkl'))
        if balanced_acc > best_balabced_acc:
            best_balabced_acc = balanced_acc
            model.save(os.path.join(modelroot, f'fold{fold}_best_balanced.pkl'))
        
        if avg_score > best_avg:
            best_avg = avg_score
            model.save(os.path.join(modelroot, f'fold{fold}_best_avg.pkl'))
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= 200:
                print(f'Early stopping at epoch {epoch}')
                break

        print(f'Epoch {epoch} [VAL] Acc: {acc:.4f} | AUC: {auc:.4f} | Balanced Acc: {balanced_acc:.4f} | Avg score: {avg_score:.4f} | Best Avg: {best_avg:.4f}')

# ============== Test ==================

def test(models, test_loader:Dataset, result_path:str):
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
                pred, _, _ = model(image, None, is_infer=True)#.numpy()
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

            
# ============== Main ==============
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='./TrainSet')
    parser.add_argument('--datacsv', type=str, default='./TrainSet/train_folds_test_thres_55_4x.csv')
    parser.add_argument('--modelroot', type=str, default='./output/convnextv1s_auxdata_thres_55_4x')
    parser.add_argument('--testonly', action='store_true', default=False)
    parser.add_argument('--loadfrom', type=str, default='./output/convnextv1s_auxdata_thres_55_4x')
    parser.add_argument('--result_path', type=str, default='./output/convnextv1s_auxdata_thres_55_4x/result.txt')
    args = parser.parse_args()

    set_seed(42) 

    if not os.path.exists(args.modelroot):
        os.makedirs(args.modelroot)

    log_path = os.path.join(args.modelroot, 'training_log.csv')
    if not os.path.exists(args.modelroot):
        os.makedirs(args.modelroot)

    if not os.path.exists(log_path):
        pd.DataFrame(columns=['fold', 'epoch', 'acc', 'auc', 'balanced_acc', 'avg_score', 'best_avg_score']).to_csv(log_path, index=False)

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
            
        #     A.OneOf([
        #     A.ElasticTransform(alpha=1, sigma=20, alpha_affine=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT,
        #                        value=0, mask_value=None, approximate=False, same_dxdy=False, p=0.5),
        #     A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT,
        #                      value=0, mask_value=None, normalized=True, p=0.5),
        # ], p=0.3),
            
            A.CoarseDropout(max_holes=6, max_height=0.15, max_width=0.25, min_holes=1, min_height=0.05, min_width=0.1,
                    fill_value=0, mask_fill_value=None, p=0.25),
                
            # A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.3, 0.3), p=0.5),
            
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
                    brightness_limit=(-0.1,0.1), 
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
        CenterCrop(448),
        ToTensor(),
        ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_test = Compose([
        ToTensor(),
        ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if not args.testonly:
        for fold in [0, 1, 2, 3, 4]:
            
            print(f"====================== Fold {fold} ======================")
            model = Net(pretrain=True, num_classes=6)
            
            df = pd.read_csv(args.datacsv)
            train_df = df[df['fold']!= fold]
            val_df = df[df['fold'] == fold]
     
            optimizer = jt.optim.AdamW(model.parameters(), lr=1e-4, eps=1e-08, betas=(0.9, 0.999), weight_decay=1e-2)
            scheduler = jt.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6, last_epoch=-1)
            
            train_loader = ImageFolder(
                root=os.path.join(args.dataroot, 'images/train'),
                df = train_df,
                transform=[transform_train1,data_transforms],
                batch_size=16,
                num_workers=8,
                shuffle=True,
                mode = "train"
            )
            val_loader = ImageFolder(
                root=os.path.join(args.dataroot, 'images/train'),
                df = val_df,
                transform=transform_val,
                batch_size=8,
                num_workers=8,
                shuffle=False,
                mode = "val"
            )
            run(model, optimizer, scheduler, train_loader, val_loader, 100, args.modelroot, fold)
    else:
        import glob
        test_loader = ImageFolder(
            root=args.dataroot,
            transform=transform_val,
            batch_size=8,
            num_workers=8,
            shuffle=False,
            mode = "val"
        )
        model_weights = glob.glob(os.path.join(args.loadfrom, 'fold*_best_acc.pkl'))
        models = []
        for weight in model_weights:
            print(weight)
            model = Net(pretrain=False, num_classes=6)
            model.load(weight)
            models.append(model)
        test(models, test_loader, args.result_path)

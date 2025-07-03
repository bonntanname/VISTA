import os
import datetime
import logging
import torch
from torch import optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from .train_transfer import get_embed, plot_training_history
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from nibabel.processing import resample_from_to
from concurrent.futures import ThreadPoolExecutor
import h5py
import hashlib
import csv
from monai.transforms import Compose, RandAffine, RandGaussianNoise, RandSpatialCrop, RandFlip, RandCoarseDropout, RandGaussianSharpen, RandAdjustContrast, Rand3DElastic, RandGridDistortion, RandHistogramShift, RandSpatialCropSamples
# === ここからget_cropped_batch_cpu関連の依存をコピペ ===
class ROICacheManager:
    """重心周囲の関心領域をキャッシュするクラス"""
    def __init__(self, cache_dir="/data1/hikari/roi_cache", roi_margin=150):
        self.cache_dir = cache_dir
        self.roi_margin = roi_margin
        os.makedirs(cache_dir, exist_ok=True)
    def get_cache_path(self, dir_number, center_world, source_affine):
        key_data = np.concatenate([center_world, source_affine.flatten()])
        hash_key = hashlib.md5(key_data.tobytes()).hexdigest()
        return os.path.join(self.cache_dir, f"{dir_number}_{hash_key}.h5")
    def extract_roi(self, ct_img, center_world):
        affine_inv = np.linalg.inv(ct_img.affine)
        center_voxel = np.dot(affine_inv[:3, :3], center_world) + affine_inv[:3, 3]
        center_voxel = np.round(center_voxel).astype(int)
        img_shape = ct_img.shape
        roi_start = np.maximum(center_voxel - self.roi_margin, 0)
        roi_end = np.minimum(center_voxel + self.roi_margin, img_shape)
        roi_start = roi_start.astype(int)
        roi_end = roi_end.astype(int)
        roi_data = ct_img.get_fdata()[roi_start[0]:roi_end[0], roi_start[1]:roi_end[1], roi_start[2]:roi_end[2]]
        roi_affine = ct_img.affine.copy()
        translation = np.dot(roi_affine[:3, :3], roi_start)
        roi_affine[:3, 3] += translation
        return roi_data, roi_affine, roi_start, roi_end
    def get_roi(self, dir_number, ct_path, center_world):
        ct_img = nib.load(ct_path, mmap=True)
        cache_path = self.get_cache_path(dir_number, center_world, ct_img.affine)
        if os.path.exists(cache_path):
            with h5py.File(cache_path, 'r') as f:
                roi_data = f['roi_data'][:]
                roi_affine = f['roi_affine'][:]
                roi_start = f['roi_start'][:]
                roi_end = f['roi_end'][:]
        else:
            roi_data, roi_affine, roi_start, roi_end = self.extract_roi(ct_img, center_world)
            with h5py.File(cache_path, 'w') as f:
                f.create_dataset('roi_data', data=roi_data)
                f.create_dataset('roi_affine', data=roi_affine)
                f.create_dataset('roi_start', data=roi_start)
                f.create_dataset('roi_end', data=roi_end)
        roi_img = nib.Nifti1Image(roi_data, roi_affine)
        return roi_img, roi_start, roi_end

def process_single_batch(roi_img, center_world, random_offset, new_voxel_size=1.5, new_shape=(128, 128, 128)):
    com_world = center_world + random_offset
    center_voxel = (np.array(new_shape) / 2) - 0.5
    new_affine = np.diag([new_voxel_size, new_voxel_size, new_voxel_size, 1])
    new_affine[:3, 3] = com_world - new_voxel_size * center_voxel
    target_img = nib.Nifti1Image(np.zeros(new_shape), new_affine)
    batch_img = resample_from_to(roi_img, target_img, order=1)
    batch_array = batch_img.get_fdata()
    batch_tensor = torch.unsqueeze(torch.from_numpy(batch_array).float(), 0)
    return batch_tensor

class LungWeightData:
    def __init__(self, csv_filename=os.path.join(os.path.dirname(__file__), "lung_weight.csv")):
        self.data = {}
        self.train = []
        self.test = []
        with open(csv_filename, newline='', encoding="utf-8") as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                dir_number = int(row["dir_number"].split("'")[1])
                com_world = (
                    float(row["com_world_0"]),
                    float(row["com_world_1"]),
                    float(row["com_world_2"])
                )
                self.data[dir_number] = com_world
                if int(row["fold"]) != 4:
                    self.train.append(dir_number)
                else:
                    self.test.append(dir_number)
    def get_com_world(self, target_dir_number):
        return self.data.get(target_dir_number, None)

lung_weight_data = LungWeightData()

def visualize_batch_slices_per_case(batch_tensors, output_dir, dir_numbers=None):
    """
    各症例ごとに1画像ファイルとしてzスライス16枚を4x4グリッドで保存。
    batch_tensors: Tensor配列 (B, 1, 128, 128, 128) またはリスト
    output_dir: 保存先ディレクトリ
    dir_numbers: 各テンソルに対応する患者IDリスト（省略不可推奨）
    """
    os.makedirs(output_dir, exist_ok=True)
    n = len(batch_tensors)
    n_grid = 16
    grid_size = 4
    for i in range(n):
        tensor = batch_tensors[i].squeeze().cpu().numpy()  # (128,128,128)
        z_slices = tensor.shape[2]
        z_idx_list = np.linspace(0, z_slices-1, n_grid, dtype=int)
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*2, grid_size*2))
        for j, z_idx in enumerate(z_idx_list):
            img = tensor[:, :, z_idx]
            ax = axes[j // grid_size, j % grid_size]
            ax.imshow(img, cmap='gray', vmin=-1000, vmax=1000)
            ax.axis('off')
            ax.set_title(f'z={z_idx}')
        for j in range(len(z_idx_list), grid_size*grid_size):
            ax = axes[j // grid_size, j % grid_size]
            ax.axis('off')
        plt.tight_layout()
        case_id = dir_numbers[i] if dir_numbers is not None else i
        plt.savefig(os.path.join(output_dir, f'id_{case_id}.png'))
        plt.close()

def get_cropped_batch_cpu(dir_number, n_batch=2, max_workers=4):
    roi_cache = ROICacheManager(roi_margin=150)
    center_world = lung_weight_data.get_com_world(dir_number)
    ct_path = f"/data1/hikari/NifTI_force_stack/{dir_number}/data.nii.gz"
    assert os.path.exists(ct_path)
    roi_img, roi_start, roi_end = roi_cache.get_roi(dir_number, ct_path, center_world)

    is_train = dir_number in lung_weight_data.train
    is_test = dir_number in lung_weight_data.test

    # augmentation pipeline（trainのみ、多め）
    aug_transform = Compose([
        RandAffine(prob=0.8, rotate_range=(0.1, 0.1, 0.1), shear_range=(0.05, 0.05, 0.05), translate_range=(5, 5, 5), scale_range=(0.1, 0.1, 0.1), mode='bilinear', padding_mode='border'),
        RandGaussianNoise(prob=0.5, mean=0.0, std=0.01),
        RandFlip(prob=0.5, spatial_axis=[0,1,2]),
        RandCoarseDropout(holes=8, spatial_size=(16,16,16), max_holes=16, fill_value=0, prob=0.7),
        RandGaussianSharpen(prob=0.3),
        RandAdjustContrast(prob=0.3, gamma=(0.7,1.5)),
        Rand3DElastic(prob=0.2, sigma_range=(2,5), magnitude_range=(1,2)),
        RandGridDistortion(prob=0.2),
        RandHistogramShift(prob=0.2, num_control_points=10),
    ])

    if is_train:
        random_offsets = [np.random.uniform(-50, 50, size=3) for _ in range(n_batch)]
    elif is_test:
        random_offsets = [np.zeros(3) for _ in range(n_batch)]
    else:
        random_offsets = [np.random.uniform(-50, 50, size=3) for _ in range(n_batch)]

    def _process(roi_img, center_world, random_offset):
        new_voxel_size = np.random.uniform(1.6, 2.0) if is_train else 1.8
        tensor = process_single_batch(roi_img, center_world, random_offset, new_voxel_size)
        if is_train:
            tensor = aug_transform(tensor)
        return tensor

    if max_workers <= 1 or n_batch <= 1:
        batch_tensors = [
            _process(roi_img, center_world, random_offset)
            for random_offset in random_offsets
        ]
    else:
        batch_tensors = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_process, roi_img, center_world, random_offset)
                for random_offset in random_offsets
            ]
            for future in futures:
                batch_tensors.append(future.result())
    return batch_tensors
# === ここまでget_cropped_batch_cpu関連 ===

# CO_value>=30か否かを予測するデータセット
class CODataset(Dataset):
    def __init__(self, dir_numbers, co_labels, device):
        assert len(dir_numbers) == len(co_labels)
        self.dir_numbers = dir_numbers
        self.co_labels = co_labels
        self.device = device
    def __len__(self):
        return len(self.dir_numbers)
    def __getitem__(self, idx):
        dir_number = self.dir_numbers[idx]
        co_label = float(self.co_labels[idx])
        image = get_cropped_batch_cpu(dir_number, 1, 1)[0]
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(co_label, dtype=torch.float32)
        return image, label

# CO_value>=30か否か分類用のモデル
class COClassifier(nn.Module):
    def __init__(self, pretrained_model, freeze_backbone=True):
        super(COClassifier, self).__init__()
        self.pretrained_model = pretrained_model
        if freeze_backbone:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(self.pretrained_model.device)
    def forward(self, x):
        features = self.pretrained_model(x)
        cls_output = self.classifier(features).squeeze(-1)
        return cls_output

def compute_bce_loss(cls_outputs, labels, device):
    criterion_cls = nn.BCELoss()
    loss_cls = criterion_cls(cls_outputs, labels)
    return loss_cls, loss_cls.item()

def plot_roc_curve_co(model, data_loader, output_dir, epoch, device='cuda'):
    from sklearn.metrics import roc_curve, auc
    roc_dir = os.path.join(output_dir, "roc")
    os.makedirs(roc_dir, exist_ok=True)
    output_name = os.path.join(roc_dir, f"roc-{epoch:04d}.png")
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.00])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_name)
    plt.close()

def plot_violin_co(model, val_loader, output_dir, epoch, device='cuda'):
    violin_dir = os.path.join(output_dir, "violin")
    os.makedirs(violin_dir, exist_ok=True)
    output_name = os.path.join(violin_dir, f"violin-{epoch:04d}.png")
    model.eval()
    val_probs = []
    val_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            val_probs.extend(outputs.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    import pandas as pd
    val_df = pd.DataFrame({
        'Probability': val_probs,
        'True Label': [f'Class {int(label)}' for label in val_labels],
    })
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    ax = sns.violinplot(x='True Label', y='Probability', 
                    data=val_df, inner='quart',
                    palette={'Class 0': 'lightblue', 'Class 1': 'lightgreen'})
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
    plt.ylim(0, 1) 
    plt.title('Distribution of Prediction Probabilities by Class (Validation Set)')
    plt.ylabel('Predicted Probability of Class 1')
    plt.xlabel('True Class')
    plt.tight_layout()
    plt.savefig(output_name)
    plt.close()

# トレーニングループ
def train_co_model(model, train_loader, val_loader, optimizer, num_epochs=10, device='cuda:0', output_dir=None):
    history = {'train_loss': [], 'val_loss': []}
    device = torch.device(device)
    model = model.to(device)
    model = nn.DataParallel(model)
    for epoch in range(num_epochs):
        if epoch == 100:
            for name, param in model.named_parameters():
                if "final.4" in name:
                    param.requires_grad = True
            logging.info(f"After unfreezing final.4 in epoch {epoch}")
            for name, param in model.named_parameters():
                logging.info(f"{name}: requires_grad={param.requires_grad}, shape={param.shape}, num_params={param.numel()}")
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            cls_outputs = model(images)
            loss, loss_cls = compute_bce_loss(cls_outputs, labels, device)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        history['train_loss'].append(avg_loss)
        logging.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                cls_outputs = model(images)
                loss, _ = compute_bce_loss(cls_outputs, labels, device)
                val_loss += loss.item() * images.size(0)
        avg_val_loss = val_loss / len(val_loader.dataset)
        history['val_loss'].append(avg_val_loss)
        logging.info(f'Validation Loss: {avg_val_loss:.4f}')
        if output_dir:
            plot_training_history(history, output_dir)
            if epoch == 0:
                # 1回目の学習ループで可視化
                batch_size = train_loader.batch_size if hasattr(train_loader, 'batch_size') else images.shape[0]
                train_dir_numbers = train_loader.dataset.dir_numbers
                for i, (images, labels) in enumerate(train_loader):
                    start = i * batch_size
                    end = start + images.shape[0]
                    visualize_batch_slices_per_case(images, os.path.join(output_dir, f"debug_batch/train"), dir_numbers=train_dir_numbers[start:end])
                    if i >= 2: break  # 2バッチだけ可視化
                batch_size = val_loader.batch_size if hasattr(val_loader, 'batch_size') else images.shape[0]
                test_dir_numbers = val_loader.dataset.dir_numbers
                for i, (images, labels) in enumerate(val_loader):
                    start = i * batch_size
                    end = start + images.shape[0]
                    visualize_batch_slices_per_case(images, os.path.join(output_dir, f"debug_batch/test"), dir_numbers=test_dir_numbers[start:end])
                    if i >= 2: break
            if epoch % 10 == 0:
                torch.save(model.state_dict(), os.path.join(output_dir, f'model-co-{epoch:04d}.pth'))
                plot_roc_curve_co(model, val_loader, output_dir, epoch, device)
                plot_violin_co(model, val_loader, output_dir, epoch, device)
    return model, history

def main2_co(output_dir):
    pretrained_model_path = '/mnt/nas/ssd/hikari/checkpoints/VISTA/SimCLR20250226/model-0770.pth'
    csv_file = "/home/hikari/MedSAM/lung_weight_co_with_choice.csv"
    batch_size = 8
    num_epochs = 301
    learning_rate = 3 * 1e-4
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    self_supervised_model = get_embed(device)
    try:
        checkpoint = torch.load(pretrained_model_path, map_location=device)
        if 'model' in checkpoint:
            self_supervised_model.load_state_dict(checkpoint['model'])
        else:
            self_supervised_model.load_state_dict(checkpoint)
        logging.info(f"Loaded pre-trained weights from {pretrained_model_path}")
    except Exception as e:
        logging.error(f"Error loading weights: {e}")
        return
    classifier = COClassifier(self_supervised_model, freeze_backbone=True)
    classifier = classifier.to(device)
    #for name, param in classifier.named_parameters():
        #if "final"in name or "bias" in name:
            #param.requires_grad = True
    for name, param in classifier.named_parameters():
        param_size = param.numel()
        logging.info(f"{name}: requires_grad={param.requires_grad}, shape={param.shape}, num_params={param_size}")
    df = pd.read_csv(csv_file)
    df = df[df['Choice'] == 'Y']
    df['dir_number'] = df['dir_number'].apply(lambda x: int(str(x).strip("'")) if isinstance(x, str) else x)
    df['co_label'] = (df['CO_value'] >= 30).astype(float)
    # foldで分割
    train_df = df[df['fold'] != 4]
    test_df = df[df['fold'] == 4]
    train_dataset = CODataset(
        dir_numbers=train_df['dir_number'].tolist(),
        co_labels=train_df['co_label'].tolist(),
        device=device
    )
    val_dataset = CODataset(
        dir_numbers=test_df['dir_number'].tolist(),
        co_labels=test_df['co_label'].tolist(),
        device=device
    )
    logging.info(f"{len(train_dataset)=},{len(val_dataset)=}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=learning_rate)
    trained_model, history = train_co_model(
        classifier,
        train_loader,
        val_loader,
        optimizer,
        num_epochs=num_epochs,
        device=device,
        output_dir=output_dir
    )
    logging.info("CO_value>=30分類モデルの学習が完了し、モデルが保存されました。")

if __name__ == "__main__":
    import shutil
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(target_dir)
    today_str = datetime.datetime.now().strftime("%Y%m%d")
    output_dir = os.path.join("/mnt/nas/ssd/hikari/checkpoints/VISTA", f"train_co_vista_with_cochoice_with_pretrained_aug_{today_str}")
    suffix = 1
    original_dir = output_dir
    while os.path.exists(output_dir):
        output_dir = f"{original_dir}_{suffix}"
        suffix += 1
    os.makedirs(output_dir)
    log_file_path = os.path.join(output_dir, "output.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )
    file_name = os.path.basename(__file__)
    shutil.copy(__file__, os.path.join(output_dir, file_name))
    main2_co(output_dir)

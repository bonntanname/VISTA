import os
import datetime
import logging
def get_checkpoint_dir(base_path, base_name):
    # 今日の日付を "YYYYMMDD" の形式で取得
    today_str = datetime.datetime.now().strftime("%Y%m%d")
    # 基本のディレクトリ名を組み立てる
    checkpoint_dir = os.path.join(base_path, f"{base_name}{today_str}")
    
    # 同名ディレクトリが存在する場合は、連番のサフィックスを追加して新たなディレクトリ名を作成
    suffix = 1
    original_dir = checkpoint_dir
    while os.path.exists(checkpoint_dir):
        checkpoint_dir = f"{original_dir}_{suffix}"
        suffix += 1
    
    return checkpoint_dir
if __name__ == '__main__' and __package__ is None:
    import sys
    # use_encoder.py の1階層上（プロジェクトのルートディレクトリ）を sys.path に追加
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # カレントディレクトリを変更
    os.chdir(target_dir)
    output_dir = get_checkpoint_dir("checkpoints","SimCLRSkull")
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
    __package__ = 'scripts'
    logging.info(f"{output_dir=}")
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import csv
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from .infer import InferClass
from monai.transforms import ScaleIntensityRange
import shutil
from concurrent.futures import ThreadPoolExecutor
import h5py
import hashlib
class ROICacheManager:
    """重心周囲の関心領域をキャッシュするクラス"""
    
    def __init__(self, cache_dir="/data1/hikari/roi_cache", roi_margin=150):
        """
        Args:
            cache_dir: キャッシュディレクトリ
            roi_margin: 重心から各方向へのマージン（ボクセル単位）
        """
        self.cache_dir = cache_dir
        self.roi_margin = roi_margin
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_path(self, dir_number, center_world, source_affine):
        """キャッシュファイルのパスを生成"""
        # 重心とアフィン行列からハッシュ値を生成
        key_data = np.concatenate([center_world, source_affine.flatten()])
        hash_key = hashlib.md5(key_data.tobytes()).hexdigest()
        return os.path.join(self.cache_dir, f"{dir_number}_{hash_key}.h5")
    
    def extract_roi(self, ct_img, center_world):
        """重心周囲のROIを抽出"""
        # 世界座標から画像座標への変換
        affine_inv = np.linalg.inv(ct_img.affine)
        center_voxel = np.dot(affine_inv[:3, :3], center_world) + affine_inv[:3, 3]
        center_voxel = np.round(center_voxel).astype(int)
        
        # 画像の範囲を取得
        img_shape = ct_img.shape
        
        # ROIの範囲を決定（画像範囲内に収める）
        roi_start = np.maximum(center_voxel - self.roi_margin, 0)
        roi_end = np.minimum(center_voxel + self.roi_margin, img_shape)
        
        roi_start = roi_start.astype(int)
        roi_end = roi_end.astype(int)
        
        # ROIを抽出
        roi_data = ct_img.get_fdata()[
            roi_start[0]:roi_end[0],
            roi_start[1]:roi_end[1],
            roi_start[2]:roi_end[2]
        ]
        
        # ROIのアフィン行列を調整
        roi_affine = ct_img.affine.copy()
        translation = np.dot(roi_affine[:3, :3], roi_start)
        roi_affine[:3, 3] += translation
        
        return roi_data, roi_affine, roi_start, roi_end
    
    def get_roi(self, dir_number, ct_path, center_world):
        """キャッシュからROIを取得するか、なければ作成してキャッシュする"""
        # 元のCT画像を読み込み（メモリマッピング使用）
        ct_img = nib.load(ct_path, mmap=True)
        
        # キャッシュパスを取得
        cache_path = self.get_cache_path(dir_number, center_world, ct_img.affine)
        
        # キャッシュが存在すればそれを読み込む
        if os.path.exists(cache_path):
            with h5py.File(cache_path, 'r') as f:
                roi_data = f['roi_data'][:]
                roi_affine = f['roi_affine'][:]
                roi_start = f['roi_start'][:]
                roi_end = f['roi_end'][:]
        else:
            # キャッシュがなければROIを抽出してキャッシュする
            roi_data, roi_affine, roi_start, roi_end = self.extract_roi(ct_img, center_world)
            
            with h5py.File(cache_path, 'w') as f:
                f.create_dataset('roi_data', data=roi_data)
                f.create_dataset('roi_affine', data=roi_affine)
                f.create_dataset('roi_start', data=roi_start)
                f.create_dataset('roi_end', data=roi_end)
        
        # nibabel形式のROIイメージを生成
        roi_img = nib.Nifti1Image(roi_data, roi_affine)
        
        return roi_img, roi_start, roi_end

def process_single_batch(roi_img, center_world, random_offset, new_shape=(128, 128, 128)):
    """
    単一バッチのリサンプリング処理を行う関数（CPU版）
    
    Args:
        roi_img: ROI画像（nibabel形式）
        center_world: 重心の世界座標
        random_offset: 適用するランダムオフセット
        new_shape: 出力画像のサイズ
        
    Returns:
        torch.Tensor: リサンプリングされたテンソル
    """
    # ランダムオフセットの適用
    com_world = center_world + random_offset
    
    # 新画像の中心ボクセル位置（0-indexedの場合）
    center_voxel = (np.array(new_shape) / 2) - 0.5
    
    # ランダムな新ボクセルサイズの決定
    new_voxel_size = np.random.uniform(1.2, 2.25)
    new_affine = np.diag([new_voxel_size, new_voxel_size, new_voxel_size, 1])
    new_affine[:3, 3] = com_world - new_voxel_size * center_voxel
    
    # 新しいターゲット画像（グリッド情報のみ）
    target_img = nib.Nifti1Image(np.zeros(new_shape), new_affine)
    
    # ROI画像をターゲットグリッドに再サンプリング（線形補間）
    batch_img = resample_from_to(roi_img, target_img, order=1)
    batch_array = batch_img.get_fdata()
    
    # NumPy配列をPyTorchのtensorに変換し、先頭にチャンネル次元を追加
    batch_tensor = torch.unsqueeze(torch.from_numpy(batch_array).float(), 0)
    
    return batch_tensor

def get_cropped_batch_cpu(dir_number, n_batch=2, max_workers=4):
    """
    CPU処理でキャッシュを活用したクロップバッチ生成関数
    
    Args:
        dir_number: ディレクトリ番号
        n_batch: 生成するバッチ数
        max_workers: 並列処理のワーカー数
        
    Returns:
        list of torch.Tensor: クロップされた画像バッチのリスト
    """
    # ROIキャッシュマネージャーの初期化
    roi_cache = ROICacheManager(roi_margin=150)
    
    # CT画像の重心位置取得（全てのbatchで共通）
    center_world = lung_weight_data.get_com_world(dir_number)
    ct_path = f"/data1/hikari/NifTI_force_stack/{dir_number}/data.nii.gz"
    assert os.path.exists(ct_path)
    
    # ROIを取得（キャッシュがあればそれを使用）
    roi_img, roi_start, roi_end = roi_cache.get_roi(dir_number, ct_path, center_world)
    
    # ランダムオフセットを事前生成
    random_offsets = [np.random.uniform(-100, 100, size=3) for _ in range(n_batch)]
    
    # 単一バッチ処理または並列処理を選択
    if max_workers <= 1 or n_batch <= 1:
        # 単一バッチまたは並列化しない場合
        batch_tensors = [
            process_single_batch(roi_img, center_world, random_offset)
            for random_offset in random_offsets
        ]
    else:
        # 並列処理でバッチを生成
        batch_tensors = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 各バッチの処理を並列で実行
            futures = [
                executor.submit(process_single_batch, roi_img, center_world, random_offset)
                for random_offset in random_offsets
            ]
            
            # 結果を収集
            for future in futures:
                batch_tensors.append(future.result())
    
    return batch_tensors
class get_embed(torch.nn.Module):
    def __init__(self, device = "cuda:0"):
        super().__init__()
        self.encoder = InferClass(config_file = os.path.join(os.path.dirname(__file__), '..', "configs/infer.yaml")).model.image_encoder.encoder
        self.device = device
        self.final = nn.Sequential(
            nn.Conv3d(768, 512, 3, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv3d(512, 256, 3, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv3d(256, 128, 3, stride = 2, padding = 1),
            nn.Flatten()
        ).to(self.device)
        self.transform = ScaleIntensityRange(a_min = -963.8247715525971, a_max = 1053.678477684517, b_min = 0.0, b_max = 1.0)
        
    def forward(self, input):
        input = self.transform(input)
        embed = self.encoder(input.to(self.device))
        ret = self.final(embed[-1])
        return F.normalize(ret, dim = 1)

class LungWeightData:
    def __init__(self, csv_filename=os.path.join(os.path.dirname(__file__), "skull_weight.csv")):
        """
        指定のCSVファイルを読み込み、各行のdir_numberをキー、com_worldを
        (float, float, float) のタプルとして辞書に格納します。
        """
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
        """
        指定のdir_numberに対応するcom_worldの値を返します。
        該当するものがない場合は None を返します。
        """
        return self.data.get(target_dir_number, None)
    

lung_weight_data = LungWeightData()

def get_cropped_batch(dir_number, n_batch=2):
    # CT画像の重心位置取得（全てのbatchで共通）
    center_world = lung_weight_data.get_com_world(dir_number)
    ct_path = f"/data1/hikari/NifTI_force_stack/{dir_number}/data.nii.gz"
    assert os.path.exists(ct_path)
    ct_img = nib.load(ct_path)
    
    new_shape = (128, 128, 128)
    # 新画像の中心ボクセル位置（0-indexedの場合）
    center_voxel = (np.array(new_shape) / 2) - 0.5

    batch_tensors = []
    for _ in range(n_batch):
        # ランダムオフセットの付与
        random_offset = np.random.uniform(-100, 100, size=3)
        com_world = center_world + random_offset

        # ランダムな新ボクセルサイズの決定
        new_voxel_size = np.random.uniform(1.2, 2.25)
        new_affine = np.diag([new_voxel_size, new_voxel_size, new_voxel_size, 1])
        new_affine[:3, 3] = com_world - new_voxel_size * center_voxel

        # 新しいターゲット画像（グリッド情報のみ）
        target_img = nib.Nifti1Image(np.zeros(new_shape), new_affine)

        # CT画像をターゲットグリッドに再サンプリング（線形補間）
        batch_img = resample_from_to(ct_img, target_img, order=1)
        batch_array = batch_img.get_fdata()

        # NumPy配列をPyTorchのtensorに変換し、先頭にチャンネル次元を追加
        batch_tensor = torch.unsqueeze(torch.from_numpy(batch_array).float(), 0)
        batch_tensors.append(batch_tensor)
    return batch_tensors
    # 各切り出し結果を連結して (n_batch, 128, 128, 128) のテンソルに
    final_tensor = torch.cat(batch_tensors, dim=0)
    return final_tensor

# 自己教師あり学習のためのデータセット
class Self3DDataset(Dataset):
    def __init__(self, data):
        self.data = data  # 5000症例のデータ
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        # 各ワーカープロセスで新しいGPUResamplerインスタンスを作成
        crops = get_cropped_batch_cpu(sample, 2, 1)
        return crops[0], crops[1]

# コントラスト損失（NT-Xent損失）
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        
        # 完全な表現行列の計算
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        # マスクの作成
        mask = torch.eye(batch_size * 2, dtype=bool, device=z_i.device)
        positives_mask = torch.roll(mask, batch_size, 1)
        negatives_mask = ~(mask | positives_mask)
        
        # ポジティブのコサイン類似度
        positives = similarity_matrix[positives_mask].view(2 * batch_size, 1)
        
        # ネガティブのコサイン類似度
        negatives = similarity_matrix[negatives_mask].view(2 * batch_size, -1)
        
        # Logitsの計算
        logits = torch.cat([positives, negatives], dim=1) / self.temperature
        
        # ラベルの作成（常に0番目がポジティブペア）
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z_i.device)
        
        return self.criterion(logits, labels)

    
def main(base_model, output_dir, num_epochs=1000, batch_size=4, lr=1e-4):
    train_dataset = Self3DDataset(lung_weight_data.train)
    test_dataset = Self3DDataset(lung_weight_data.test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    optimizer = optim.Adam(base_model.parameters(), lr=lr)
    criterion = NTXentLoss(temperature=0.1)
    train_losses = []
    test_losses = []
    logging.info(f"{len(train_dataset)=},{len(test_dataset)=}")


    for epoch in range(num_epochs):
        # Training
        base_model.train()
        total_train_loss = 0
        for (crop1, crop2) in train_dataloader:
            z_i = base_model(crop1)
            z_j = base_model(crop2)
            # 損失の計算
            loss = criterion(z_i, z_j)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        if epoch % 10 == 0:
            torch.save(base_model.state_dict(), os.path.join(output_dir, f"model-{epoch:04d}.pth"))
        # Validation
        base_model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for (crop1, crop2) in test_dataloader:
                z_i = base_model(crop1)
                z_j = base_model(crop2)
                # 損失の計算
                loss = criterion(z_i, z_j)
                total_test_loss += loss.item()
        avg_test_loss = total_test_loss / len(test_dataloader)
        test_losses.append(avg_test_loss)

        logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
        epochs = range(1, len(train_losses) + 1)

        # Plot the linear scale loss
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, test_losses, label='Test Loss')
        plt.title('Train and Test Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(output_dir, "loss.png"))
        plt.close()

class DrownClassifier(nn.Module):
    def __init__(self, pretrained_model, freeze_backbone=True):
        super(DrownClassifier, self).__init__()
        self.pretrained_model = pretrained_model
        
        # 事前学習済みモデルの重みを凍結するかどうか
        if freeze_backbone:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
        
        # 分類ヘッド
        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()  # バイナリ分類なのでシグモイド活性化
        ).to(self.pretrained_model.device)
    
    def forward(self, x):
        # 特徴抽出（プロジェクションヘッドは使用しない）
        features = self.pretrained_model(x)
        # 分類
        output = self.classifier(features)
        return output.squeeze()  # バッチ次元以外を潰す
# CSVからdrownデータを読み込むためのクラス
class DrownDataset(Dataset):
    def __init__(self, dir_numbers, drown_labels, device):
        """
        Args:
            dir_numbers (list): dir_number のリスト
            drown_labels (list): drown ラベルのリスト (0または1)
            transform (callable, optional): データに適用する変換
        """
        assert len(dir_numbers) == len(drown_labels), "dir_numbersとdrown_labelsの長さが一致しません"
        
        self.dir_numbers = dir_numbers
        self.drown_labels = drown_labels
        
        # クラス重みを計算
        from collections import Counter
        self.class_counts = Counter(drown_labels)
        print(f"クラス分布: {dict(self.class_counts)}")
        
    def __len__(self):
        return len(self.dir_numbers)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        dir_number = self.dir_numbers[idx]
        drown_label = float(self.drown_labels[idx])
        
        # get_cropped_batch関数を使用して画像データを取得
        image = get_cropped_batch_cpu(dir_number,1,1)[0]
        
        # テンソルに変換
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(drown_label, dtype=torch.float32)
        
        
        return image, label
# Violin Plotを描画する関数
def plot_violin(model, val_loader, output_dir, epoch, device='cuda'):
    """検証データのクラスごとの予測確率分布をViolin Plotで可視化する関数"""
    import seaborn as sns
    output_name = os.path.join(output_dir, "violin", f"violin-{epoch:04d}.png")
    model.eval()
    val_probs = []
    val_labels = []
    
    # 検証セットの予測
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            val_probs.extend(outputs.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    # データフレームの作成
    import pandas as pd
    
    val_df = pd.DataFrame({
        'Probability': val_probs,
        'True Label': [f'Class {int(label)}' for label in val_labels],
    })
    
    # Violin Plotの作成
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    ax = sns.violinplot(x='True Label', y='Probability', 
                    data=val_df, inner='quart',
                    palette={'Class 0': 'lightblue', 'Class 1': 'lightgreen'})
    
    # 決定境界（0.5）を点線で示す
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
    
    plt.title('Distribution of Prediction Probabilities by Class (Validation Set)')
    plt.ylabel('Predicted Probability of Class 1')
    plt.xlabel('True Class')
    plt.tight_layout()
    plt.savefig(output_name)
    plt.close()
    
def plot_roc_curve(model, data_loader, output_dir, epoch, device='cuda'):
    """モデルのROC曲線をプロットする関数"""
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    output_name = os.path.join(output_dir, "roc", f"roc-{epoch:04d}.png")
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
    
    # ROC曲線
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # PR曲線
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)
    
    # 結果をプロット
    plt.figure(figsize=(12, 5))
    
    # ROC曲線
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # PR曲線
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label=f'AP = {pr_auc:.3f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(output_name)
    plt.close()

# 学習曲線のプロット
def plot_training_history(history, output_dir):
    """トレーニング中の損失と精度の履歴をプロットする関数"""
    """トレーニング中の損失と精度の履歴をプロットする関数"""
    plt.figure(figsize=(12, 4))
    
    # 損失のプロット
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 精度のプロット
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.plot(history['val_auc'], label='Validation AUC')
    plt.title('Metrics over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,'training_history.png'))
    plt.close()
# トレーニングループ
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda:1'):
    """
    モデルのトレーニングと評価を行う関数
    
    Args:
        model: トレーニングするモデル
        train_loader: トレーニングデータのDataLoader
        val_loader: 検証データのDataLoader
        criterion: 損失関数
        optimizer: オプティマイザ
        num_epochs: エポック数
        device: 'cuda'または'cpu'
    
    Returns:
        トレーニング済みのモデルと、トレーニング中の損失と精度の履歴
    """
    # 結果を記録するための辞書
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_auc': []
    }
    
    # デバイスの設定
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # AUCの計算用
    from sklearn.metrics import roc_auc_score
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # トレーニングフェーズ
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 勾配をゼロにリセット
            optimizer.zero_grad()
            
            # フォワードパス
            outputs = model(inputs)
            
            # 損失の計算
            loss = criterion(outputs, labels)
            
            # バックワードパスと最適化
            loss.backward()
            optimizer.step()
            
            # 統計の更新
            running_loss += loss.item() * inputs.size(0)
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct / total
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        
        # 検証フェーズ
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # フォワードパス
                outputs = model(inputs)
                
                # 損失の計算
                loss = criterion(outputs, labels)
                
                # 統計の更新
                running_loss += loss.item() * inputs.size(0)
                predicted = (outputs >= 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # AUC計算用にデータを保存
                all_outputs.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total
        
        # AUCの計算
        try:
            val_auc = roc_auc_score(all_labels, all_outputs)
        except:
            val_auc = 0.5  # エラーが発生した場合のデフォルト値
            
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['val_auc'].append(val_auc)
        
        logging.info(f'Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}')
        logging.info(f'Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f} AUC: {val_auc:.4f}')
        plot_training_history(history, output_dir)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(output_dir,f'model-{epoch:04d}.pth'))
            plot_roc_curve(model, val_loader, output_dir, epoch, device)
            plot_violin(model, val_loader, output_dir, epoch, device)
    
    return model, history


# メイン実行関数
def main2(output_dir):
    # 設定
    #pretrained_model_path = '/home/hikari/VISTA/vista3d/checkpoints/SimCLR20250226/model-0770.pth'
    csv_file = '/home/hikari/VISTA/vista3d/scripts/lung_weight_with_drown.csv'
    batch_size = 4
    num_epochs = 101
    learning_rate = 5 * 1e-4
    
    # デバイスの設定
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # 基盤モデルの作成
    self_supervised_model = get_embed(device)
    
    # 事前学習済みの重みをロード
    #try:
        #checkpoint = torch.load(pretrained_model_path, map_location=device)
        ## チェックポイントがモデルの状態辞書そのものか、状態辞書を含む辞書かによって処理を分ける
        #if 'model' in checkpoint:
            #self_supervised_model.load_state_dict(checkpoint['model'])
        #else:
            #self_supervised_model.load_state_dict(checkpoint)
        #logging.info(f"Loaded pre-trained weights from {pretrained_model_path}")
    #except Exception as e:
        #logging.error(f"Error loading weights: {e}")
        #return
    
    # 分類モデルの作成
    classifier = DrownClassifier(self_supervised_model, freeze_backbone=True)
    classifier = classifier.to(device)
    
    df = pd.read_csv(csv_file)
    df['dir_number'] = df['dir_number'].apply(
        lambda x: int(str(x).strip("'")) if isinstance(x, str) else x
    )
    # データセットの作成
    train_df = df[df['dir_number'].isin(lung_weight_data.train)]
    test_df = df[df['dir_number'].isin(lung_weight_data.test)]
    
    # トレーニングデータセットとテストデータセットを作成
    train_dataset = DrownDataset(
        dir_numbers=train_df['dir_number'].tolist(),
        drown_labels=train_df['drown'].tolist(),
        device=device
    )
    
    val_dataset = DrownDataset(
        dir_numbers=test_df['dir_number'].tolist(),
        drown_labels=test_df['drown'].tolist(),
        device=device
    )
    
    logging.info(f"{len(train_dataset)=},{len(val_dataset)=}")
    
    # データローダーの作成
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # クラス重みの計算
    
    # 損失関数とオプティマイザの設定
    # 重み付きバイナリクロスエントロピー損失
    criterion = nn.BCELoss(weight=torch.tensor(9.0)).to(device)  # クラス1（溺死）の重みを使用
    
    # オプティマイザの設定
    # 凍結されていない層のみを最適化
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=learning_rate)
    
    # モデルのトレーニング
    trained_model, history = train_model(
        classifier, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        num_epochs=num_epochs, 
        device=device
    )
    
    # モデルの保存
    logging.info("Training completed and model saved.")

    

if __name__ == "__main__":
    device = "cuda:0"
    base_model = get_embed(device)

    for name, param in base_model.encoder.named_parameters():
        if name.endswith("bias"):
            param.requires_grad=True
        else:
            param.requires_grad=False
    for name, param in base_model.named_parameters():
        param_size = param.numel()
        logging.info(f"{name}: requires_grad={param.requires_grad}, shape={param.shape}, num_params={param_size}")
    file_name = os.path.basename(__file__)
    shutil.copy(__file__, os.path.join(output_dir, file_name))
    main(base_model, output_dir)


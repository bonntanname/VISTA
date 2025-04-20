import nibabel as nib
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import multiprocessing
def extract_region_info(nii_file_path, id):
    # NIfTIファイルの読み込み
    nii = nib.load(nii_file_path)
    data = nii.get_fdata()

    # 値が1のボクセルのインデックスを取得
    indices = np.where(data == 1)

    if len(indices[0]) == 0:
        print(f"No '1' values found in the file: {nii_file_path}")
        x_min = -1
        x_max = -1
        y_min = -1
        y_max = -1
        z_min = -1
        z_max = -1
    else:
        # 各軸の最小値と最大値を取得
        x_min, x_max = indices[0].min(), indices[0].max()
        y_min, y_max = indices[1].min(), indices[1].max()
        z_min, z_max = indices[2].min(), indices[2].max()

    # 配列の形状を取得
    shape = data.shape

    # 結果を辞書形式で保存
    result = {
        'filename': id,
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'z_min': z_min,
        'z_max': z_max,
        'shape_x': shape[0],
        'shape_y': shape[1],
        'shape_z': shape[2],
        'voxels' : np.count_nonzero(data)
    }
    print(f"processed {id}")
    return result

def unpack_extract_region_info(args):
    file_path, id = args
    return extract_region_info(file_path, id)

def process_nii_files(nii_name):
    output_filename = f"{nii_name}_voi_data.csv"
    if os.path.isfile(output_filename):
        print(f"{output_filename} already exists.")
        return
    seg_dir = "/data1/hikari/totalsegmentator/"
    results = []
    file_names = []
    for id in os.listdir(seg_dir):
        file_path = os.path.join(seg_dir, id, "data", "segmentations", f"{nii_name}.nii.gz")
        file_names.append((file_path, id))
    

    with multiprocessing.Pool(processes = 8) as pool:
        # 各タプルを個別の引数として展開
        results = list(pool.map(unpack_extract_region_info, file_names))
    #for id in os.listdir(seg_dir):
        #file_path = os.path.join(seg_dir, id, "data", "segmentations", f"{nii_name}.nii.gz")
        #info = extract_region_info(file_path, id)
        #if info:
            #results.append(info)

    if results:
        # pandas DataFrame に変換
        df = pd.DataFrame(results)
        # CSV に出力
        df.to_csv(output_filename, index=False)
        print(f"CSVファイルが正常に保存されました")
    else:
        print("処理するデータがありませんでした。")

if __name__ == "__main__":
    process_nii_files("body")
    #for nii_name in os.listdir("/data1/hikari/totalsegmentator/9870/data/segmentations"):
        #name = nii_name.split('.')[0]
        #print(f"{nii_name=}, {name=}",flush=True)
        #process_nii_files(name)

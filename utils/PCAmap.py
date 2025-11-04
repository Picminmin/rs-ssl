

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
def pcamap(
    X,
    save_dir,
    pc_num = 1, # 第1主成分のPCAマップを表示
    n_components = 20,
    save_flag = False
):
    """
    第pc_num主成分のPCAマップを表示する
    PCAには平均中心化と白色化whiteningを施す
    Args:
        X (_type_): _description_
        dataset_keyword (_type_): _description_
        save_path (_type_): _description_
        pc_num (int, optional): _description_. Defaults to 1.
    """
    H, W, B = X.shape
    X_flat = X.reshape(-1, B)
    X_mean = np.mean(X_flat, axis=0)
    X_centered = X_flat - X_mean
    pca = PCA(n_components=n_components, whiten=True)
    X_pca = pca.fit_transform(X_centered).reshape(H, W, n_components)
    band = X_pca[:, :, pc_num - 1]
    band = (band - band.min()) / (band.max() - band.min())
    plt.figure(figsize=(6, 5))
    plt.imshow(band, cmap='jet')
    plt.title(f'Principal Component {pc_num} Score Map')
    plt.axis("off")
    plt.colorbar(label='Normalized pca score')
    save_path = os.path.join(save_dir, f'PCA_component_{pc_num}.png')
    if save_flag:
        plt.savefig(save_path, bbox_inches = 'tight', dpi = 300)
    plt.show()
    plt.close()

if __name__ == '__main__':
    from RS_GroundTruth.rs_dataset import RemoteSensingDataset
    dataset_keyword = 'Indianpines'
    ds = RemoteSensingDataset(remove_bad_bands = True)
    X, y = ds.load(dataset_keyword)
    category_num = ds.category_num(dataset_keyword)
    pcamap(
        X = X,
        save_dir = "img",
        pc_num = 2,
        n_components = 20,
        save_flag = False
    )

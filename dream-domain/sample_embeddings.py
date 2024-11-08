import os, sys
import numpy as np
import torch
import torch.nn as nn
import faiss

sys.path.append(os.path.expandvars('$WORK/dream-domain/code/dream-ood-main'))
from scripts.KNN import generate_outliers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 100

anchor = torch.from_numpy(np.load('dream-ood-main/token_embed_in100.npy'))
id_features = torch.tensor(np.load('../results/encoder/id_feat_in100.npy')).cuda()
ID = nn.functional.normalize(id_features, p=2, dim=2)
K_in_knn = 300
gaussian_mag_ood_det = 0.03
gaussian_mag_ood_gene = 0.01
ood_gene_select = 1000
ood_det_select = 200
res = faiss.StandardGpuResources()
KNN_index = faiss.GpuIndexFlatL2(res, 768)
new_dis = torch.distributions.MultivariateNormal(torch.zeros(768).cuda(), torch.eye(768).cuda())

for c in range(num_classes):
    print(c)

    ID_class = ID[c]
    negative_samples = new_dis.rsample((1500,))
    for index in range(100):
        sample_point, boundary_point = generate_outliers(ID_class,
                                                        input_index=KNN_index,
                                                        negative_samples=negative_samples,
                                                        ID_points_num=1,
                                                        K=K_in_knn,
                                                        select=ood_gene_select,
                                                        cov_mat=gaussian_mag_ood_gene,
                                                        sampling_ratio=1.0,
                                                        pic_nums=100,
                                                        depth=768, 
                                                        shift=1)

        sample_point_in = torch.cat([sample_point_in, sample_point], 0) if index > 0 else sample_point
            
    for index in range(100):
        sample_point, boundary_point = generate_outliers(ID_class,
                                                            input_index=KNN_index,
                                                            negative_samples=negative_samples,
                                                            ID_points_num=2,
                                                            K=K_in_knn,
                                                            select=ood_det_select,
                                                            cov_mat=gaussian_mag_ood_det, 
                                                            sampling_ratio=1.0, 
                                                            pic_nums=50,
                                                            depth=768,
                                                            shift=0)
        sample_point_out = torch.cat([sample_point_out, sample_point], 0) if index > 0 else sample_point


    if c == 0:
        inliers_samples = [sample_point_in * anchor[index].norm()]
        outliers_samples = [sample_point_out * anchor[index].norm()]
    else:
        inliers_samples.append(sample_point_in * anchor[index].norm())
        outliers_samples.append(sample_point_out * anchor[index].norm())

np.save \
    ('../results/embeddings/inlier_npos_embed'+ '_noise_' + str(gaussian_mag_ood_gene)  + '_select_'+ str(
    ood_gene_select) + '_KNN_'+ str(K_in_knn) + '.npy', torch.stack(inliers_samples).cpu().data.numpy())
np.save \
    ('../results/embeddings/outlier_npos_embed' + '_noise_' + str(gaussian_mag_ood_det) + '_select_' + str(
    ood_det_select) + '_KNN_' + str(K_in_knn) + '.npy', torch.stack(outliers_samples).cpu().data.numpy())
# PCA-face-recognation
# encoding=utf-8
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from PIL import Image
file_dir = "D:\Documents\Downloads\CroppedYale"
os.chdir(file_dir)
# In[1]:
def Matrix_to_image(M,H,W,cols=10,scale=1):
    m = np.shape(M)[0]
    rows = int(math.ceil((m+0.0)/cols))
    plt.figure(1,figsize=[scale*20.0/H*W,scale*20.0/cols*rows],dpi=300)
    for i in range(m):
        plt.subplot(rows,cols,i+1)
        plt.imshow(np.reshape(M[i,:],[H,W]), cmap = plt.get_cmap("gray"))
        plt.axis('off')
# In[2]:
def pca(D):
    num_data,dim = D.shape#152 32256
    M = np.dot(D,D.T) 
    e,EV = np.linalg.eigh(M) 
    tmp = np.dot(D.T,EV).T 
    V = tmp[::-1] 
    S = np.sqrt(abs(e))[::-1]
    for i in range(V.shape[1]):
        V[:,i] /= S
    return V,S    # 返回投影矩阵、方差
# In[3]:
def filenames_process(file_dir, train_list):
    dir_list = os.listdir(file_dir)
    file_list = []
    for dir in dir_list:
        for view in train_list:
            filename = "%s/%s_%s.pgm" % (dir, dir, view)
            file_list.append(filename)
    return(file_list)
# In[4]:
train_list = ['P00A+000E+00', 'P00A+005E+10' , 'P00A+005E-10' , 'P00A+010E+00']

file_list = filenames_process(file_dir, train_list)

im = Image.open(file_list[0]).convert("L")
H,W = np.shape(im)#192 168

im_number = len(file_list)#152
arr = np.zeros([im_number,H*W],dtype=np.float32)

for i in range(im_number):
	im = Image.open(file_list[i]).convert("L")
	arr[i, : ] = np.reshape(np.asarray(im),[1,H*W])
Matrix_to_image(arr[:16],H,W,cols=4)#显示前16个原始人脸

# In[5]:
mean_image = np.mean(arr, axis=0)#显示平均脸
plt.imshow(np.reshape(mean_image,[H,W]), cmap = plt.get_cmap("gray"))
plt.axis('off')
#数据中心化
arr_norm = np.zeros([im_number, H*W])
arr_norm = arr - mean_image

# In[6]:
# 应用PCA
V,S = pca(arr_norm)
eigenfaces = V[:im_number]
Matrix_to_image(eigenfaces[:16], H, W,cols=4)#显示特征脸

# In[7]:
#绘制主成分解释的方差图
explained_var_ = (S ** 2) 
total_var = explained_var_.sum()
pve = explained_var_ / total_var

plt.plot(range(len(pve)), pve,color='red')
plt.axis([0, 160, 0, 0.4])

plt.title("variance")
plt.ylabel("Proportion of variance Explained", fontsize=12)
plt.xlabel("Principal Component Number", fontsize=12)

# In[8]:
#特征脸重建
img_idx = file_list.index('yaleB01/yaleB01_P00A+010E+00.pgm')
loadings = V[:im_number]
n_components = loadings.shape[0]#152
scores = np.dot(arr_norm[:,:], loadings[:,:].T)

img_proj = []
for m in range(n_components):
    proj = np.dot(scores[img_idx, m], loadings[m,:])
    img_proj.append(proj)
    
faces = mean_image
face_list = []
face_list.append(mean_image)
for i in range(len(img_proj)):
    faces = np.add(faces, img_proj[i])
    face_list.append(faces)
face_arr = np.asarray(face_list)
Matrix_to_image(face_arr[:16], H, W,cols=6)#显示特征脸

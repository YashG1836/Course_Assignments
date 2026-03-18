#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#
#                                   ES335- Machine Learning- Assignment 1
#
# This file is used to create the dataset for the mini-project. The dataset is created by reading the data from
# the Combined folder. The data is then split into training, testing, and validation sets. This split is supposed
# to be used for all the modeling purposes.
#
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# Library imports
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

# Constants
time = 10
offset = 100
folders = ["LAYING","SITTING","STANDING","WALKING","WALKING_DOWNSTAIRS","WALKING_UPSTAIRS"]
classes = {"WALKING":1,"WALKING_UPSTAIRS":2,"WALKING_DOWNSTAIRS":3,"SITTING":4,"STANDING":5,"LAYING":6}
id2class = {v: k for k, v in classes.items()}

combined_dir = os.path.join("Combined")

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                                                # Train Dataset
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

X_train=[]
y_train=[]
dataset_dir = os.path.join(combined_dir,"Train")

for folder in folders:
    files = os.listdir(os.path.join(dataset_dir,folder))

    for file in files:

        df = pd.read_csv(os.path.join(dataset_dir,folder,file),sep=",",header=0)
        df = df[offset:offset+time*50]
        X_train.append(df.values)
        y_train.append(classes[folder])

X_train = np.array(X_train)
y_train = np.array(y_train)


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                                                # Test Dataset
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

X_test=[]
y_test=[]
dataset_dir = os.path.join(combined_dir,"Test")

for folder in folders:
    files = os.listdir(os.path.join(dataset_dir,folder))
    for file in files:

        df = pd.read_csv(os.path.join(dataset_dir,folder,file),sep=",",header=0)
        df = df[offset:offset+time*50]
        X_test.append(df.values)
        y_test.append(classes[folder])

X_test = np.array(X_test)
y_test = np.array(y_test)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                                                # Final Dataset
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# USE THE BELOW GIVEN DATA FOR TRAINING and TESTING purposes

# concatenate the training and testing data
X = np.concatenate((X_train,X_test))
y = np.concatenate((y_train,y_test))

# split the data into training and testing sets. Change the seed value to obtain different random splits.
seed = 4
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=seed,stratify=y)

print("Training data shape: ",X_train.shape)
print("Testing data shape: ",X_test.shape)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# Solving

# Cell 2: Q1 – waveform plots

import matplotlib.pyplot as plt

T = X_train.shape[1]
D = X_train.shape[2]
print("T =", T, "D =", D)

fig, axes = plt.subplots(1, 6, figsize=(20, 3), sharex=True, sharey=True)

activity_ids = sorted(np.unique(y_train))   # [1,2,3,4,5,6]

for i, cls_id in enumerate(activity_ids):
    # pick the first example of this class
    idx = np.where(y_train == cls_id)[0][0]
    sample = X_train[idx]        # shape (T, D)

    t_axis = np.arange(T)

    # assume first 3 columns are acc_x, acc_y, acc_z
    axes[i].plot(t_axis, sample[:, 0], label='acc_x')
    axes[i].plot(t_axis, sample[:, 1], label='acc_y')
    axes[i].plot(t_axis, sample[:, 2], label='acc_z')

    axes[i].set_title(id2class[cls_id])
    axes[i].set_xlabel("Time (samples)")
    if i == 0:
        axes[i].set_ylabel("Acceleration")

fig.suptitle("Accelerometer waveforms for one sample of each activity", y=1.05)
axes[0].legend(loc='upper right')
plt.tight_layout()
plt.show()


# Cell 3: Q2 – static vs dynamic using total acceleration

import seaborn as sns
import numpy as np

# total acceleration time series per sample: shape (N, T)
acc_total = np.sqrt(
    X_train[:, :, 0]**2 +
    X_train[:, :, 1]**2 +
    X_train[:, :, 2]**2
)

# summarize each sample by its mean total acceleration
mean_total = acc_total.mean(axis=1)   # shape (N,)

df_summary = pd.DataFrame({
    "mean_acc_total": mean_total,
    "label": y_train,
    "activity": [id2class[i] for i in y_train]
})

plt.figure(figsize=(8,4))
sns.boxplot(data=df_summary, x="activity", y="mean_acc_total")
plt.xticks(rotation=45)
plt.title("Mean total acceleration by activity")
plt.tight_layout()
plt.show()


# Cell 4: Q3 – PCA on total acceleration time series

from sklearn.decomposition import PCA

# acc_total already computed: shape (N_train, T)
X_total = acc_total   # rename for clarity

pca_total = PCA(n_components=2)
X_total_pca = pca_total.fit_transform(X_total)   # shape (N_train, 2)

plt.figure(figsize=(6,5))
scatter = plt.scatter(X_total_pca[:,0], X_total_pca[:,1],
                      c=y_train, cmap='tab10', alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA on total acceleration (time series → 2D)")
plt.colorbar(scatter, label="Activity ID")
plt.tight_layout()
plt.show()


# Cell 5: Q4 – TSFEL features + PCA

import tsfel

# configuration: you can also restrict to one domain (e.g. 'statistical')
cfg = tsfel.get_features_by_domain()

# We will extract features on total acceleration time series for each sample
# acc_total: shape (N_train, T)

features_list = []
for i in range(X_train.shape[0]):
    series = acc_total[i]  # 1D array of length T
    df_ts = pd.DataFrame({"acc_total": series})
    feats = tsfel.time_series_features_extractor(cfg, df_ts, verbose=0)
    features_list.append(feats.values.flatten())

features_tsfel = np.vstack(features_list)  # shape (N_train, n_features_tsfel)

# PCA on TSFEL features
pca_tsfel = PCA(n_components=2)
X_tsfel_pca = pca_tsfel.fit_transform(features_tsfel)

plt.figure(figsize=(6,5))
scatter = plt.scatter(X_tsfel_pca[:,0], X_tsfel_pca[:,1],
                      c=y_train, cmap='tab10', alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA on TSFEL features")
plt.colorbar(scatter, label="Activity ID")
plt.tight_layout()
plt.show()


# Packages

import numpy as np
from multiprocessing import Pool, current_process
import time
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.cluster import KMeans,MiniBatchKMeans
import pandas as pd
import os,glob,shutil,random,pickle
from tqdm import tqdm
import random
from sklearn.cluster import KMeans,MiniBatchKMeans
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer


# Load CNN and generate features 

def image_feature(direc):
    model = InceptionV3(weights='imagenet',include_top=False)
    features = []
    img_name = []
    for i in tqdm(direc):
        fname = dir_imag+'/'+i
        img = image.load_img(fname,target_size=(299,299))
        x = img_to_array(img)
        x = np.expand_dims(x,axis=0)
        x = preprocess_input(x)
        feat = model.predict(x)
        feat = feat.flatten()
        features.append(feat)
        img_name.append(i)
    return features,img_name


# Determine number of clusters

def det_clusters(img_features):
    Max_clust = 11
    model5 = MiniBatchKMeans()
    plt.figure()
    visualizer = KElbowVisualizer(model5, k =(2,Max_clust),timings=True,metric='calinski_harabasz')
    visualizer.fit(np.array(img_features))
    visualizer.show()

# Train the minibacthkmeans using 10,000 random images
# This will not run because I did not upload all the images
def train_dataset():
     rand_images = 10000
     size_batch = rand_images//10
     img_path = os.listdir(dir_imag)
     rand_list_imgs = random.choices(img_path,k= rand_images)
     img_features,img_name=image_feature(rand_list_imgs)
     image_cluster = pd.DataFrame(img_name,columns=['image'])
     # This number is determined with the previous function
     n_clust = 2
     kmeans_try = MiniBatchKMeans(n_clusters=n_clust, init='k-means++',batch_size=size_batch)
     kmeans_try.fit(img_features)
     # Save trained model into a .pkl file 
     pickle.dump(kmeans_try , open("miniba_trained_model.pkl", "wb"))

    

dir_imag = 'final_images'
# Load trained clusters based on 10,000 random images
kmeans_trained = pickle.load(open('miniba_trained_model.pkl','rb'))
img_path = os.listdir(dir_imag)

# Generate folders to move scalograms to each cluster
if not os.path.exists('clust_img/cluster0'):
     os.makedirs('clust_img/cluster0')

if not os.path.exists('clust_img/cluster1'):
     os.makedirs('clust_img/cluster1')


img_features,img_name = image_feature(img_path)
# Determine number of clusters and plot it
det_clusters(img_features)

# Predict the cluster for each image contained in img_path
predictions = kmeans_trained.predict(img_features)
image_cluster = pd.DataFrame(img_name,columns=['image'])
image_cluster["clusterid"] = predictions
dummy = image_cluster['image'].str.split('.',expand=True,n=3)
image_cluster['stations'] = dummy[0]
image_cluster['Time'] = dummy[1]
# Move original scalograms to their respective cluster
for iix in range(len(image_cluster)):
    shutil.copy(os.path.join(dir_imag, image_cluster['image'][iix]),
                'clust_img/cluster'+str(image_cluster['clusterid'][iix])+'/'+str(image_cluster['image'][iix]))
# Save a csv file that identify which scalogram belong to which cluster
image_cluster.to_csv('total_clusts.csv')








# DAS_ice
Here I provide a minimal example code to generate features using InceptionV3, load the trained model and classify some scalograms. 
There are some packages that need to be installed in order to run the code: numpy, tensorflow, scikit-learn, tqdm, yellowbrick, etc.
All functions are inside "run_all.py". 
final_images contain scalograms for the entire two weeks for two channnels: 00350 (distance = 0.59 km) and 15550 (dist = 31.62 km)
python run_all.py should :
  (i) generate a graphic showing that there are two clusters for the scalograms in final_images according with the
      Calinski Index. 
 (ii) generature features for each scalogram, recognize if scalogram belong to cluster 0 or 1 based on those features and on the trained model (.pkl file).
      clust_img will contain the scalograms for each cluster and the results are saved  to "total_clusts.csv" 

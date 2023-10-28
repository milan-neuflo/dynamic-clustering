import time
import requests
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import warnings
import os
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)

def get_new_data():
    response = requests.get('http://localhost:5000/stream-data')
    if response.status_code == 200:
        json_response = response.json()
        return pd.DataFrame(json_response["data"]), json_response["stop"]
    else:
        return pd.DataFrame(), False

data = pd.DataFrame()
print("DataFrame initialized successfully.")

model_iteration = 1
max_iterations = 50  # Maximum number of iterations

if not os.path.exists('elbow_graphs'):
    os.makedirs('elbow_graphs')

if not os.path.exists('clusters'):
    os.makedirs('clusters')

scaler = StandardScaler()
pca = PCA(n_components=2)

# List to store silhouette scores
silhouette_scores = []

def optimal_num_clusters(data):
    sse = []
    list_k = list(range(1, 6))

    for k in list_k:
        km = MiniBatchKMeans(n_clusters=k)
        km.fit(data)
        sse.append(km.inertia_)

    diff_sse = np.diff(sse)
    elbow_point = np.argwhere(diff_sse == max(diff_sse))
    optimal_k = list_k[int(elbow_point)]
    return optimal_k

while model_iteration <= max_iterations:
    new_data, stop = get_new_data()
    
    if stop:
        print("Received stop signal from server. Stopping data collection and processing.")
        break
    
    if not new_data.empty:
        print(f"Received {len(new_data)} new data points.")  

    data = pd.concat([data, new_data], ignore_index=True)
    print("Total data size: ", data.shape)

    if data.shape[0] % 200 == 0 and data.shape[0] != 0:
        scaled_data = scaler.fit_transform(data)
        
        # Perform PCA for visualization
        reduced_data = pca.fit_transform(scaled_data)

        optimal_k = optimal_num_clusters(scaled_data)
        model = MiniBatchKMeans(n_clusters=optimal_k)
        model.fit(scaled_data)

        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=model.labels_, cmap='viridis')
        plt.savefig(f'clusters/cluster_{model_iteration}.png')
        plt.close()

        silhouette_avg = silhouette_score(scaled_data, model.labels_)
        print(f"For n_clusters = {optimal_k}, The average silhouette_score is: {silhouette_avg}")
        
        # Append the silhouette score to the list
        silhouette_scores.append(silhouette_avg)

        # Elbow method visualization
        visualizer = KElbowVisualizer(model, k=(2,10))
        visualizer.fit(scaled_data)
        visualizer.show(outpath=f"elbow_graphs/elbow_{model_iteration}.png")
        plt.close()

        model_iteration += 1
        print(f"Training cycle {model_iteration} completed. Number of clusters: {optimal_k}. Silhouette score: {silhouette_avg}.")
        
    time.sleep(0.2)

if model_iteration > max_iterations:
    print("Reached max number of iterations. Stopping data collection and processing.")
# After the main loop, plot the silhouette scores
plt.plot(range(1, len(silhouette_scores) + 1), silhouette_scores, marker='o')
plt.title('Iterations vs Silhouette Score')
plt.xlabel('Iteration')
plt.ylabel('Silhouette Score')
plt.grid(True)

# Save the plot as an image file
plt.savefig('silhouette_scores_plot.png')
plt.close()
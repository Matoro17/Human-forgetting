import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

# Carregar features e labels
features = np.load("features.npy")
labels = np.load("labels.npy")

# Redução para 3D com t-SNE
tsne = TSNE(n_components=3, random_state=42)
features_3d = tsne.fit_transform(features)

# Plotar
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    features_3d[:, 0], features_3d[:, 1], features_3d[:, 2],
    c=labels, cmap='tab10', s=10, alpha=0.7
)
legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend1)

plt.show()

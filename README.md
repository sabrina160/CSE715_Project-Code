# Multi-modal Beta-VAEs for Uncovering the Latent Structure of Music through Disentangled Representation and Interpretable Clustering
This project implements a progressive Variational Autoencoder (VAE) framework for uncovering latent structures in large-scale music data using audio features, lyrics, and genre information. The goal is to learn disentangled, interpretable representations that improve clustering quality in multi-modal music datasets.

# Overview
Traditional clustering methods struggle with music data due to:
- Nonlinear feature relationships
- Multi-modal complexity (audio + lyrics)
- Poor latent representation

To address this, this project proposes a 3-stage progressive learning pipeline:
- Basic VAE → learns latent representation from audio
- Convolutional VAE (ConvVAE) → captures local audio structures and enables multimodal fusion with lyrics
- Conditional / Beta-VAE (CVAE) → learns disentangled multimodal representations with genre conditioning

The final model (Conditional / Beta-VAE (CVAE)) integrates:
- Audio features
- Lyrics embeddings
- Genre labels (as conditioning)

# Key Contributions
- Progressive VAE pipeline for music clustering
- Multi-modal fusion of audio + lyrics
- Disentangled latent representation using β-VAE
- Comprehensive clustering evaluation across multiple methods
- Extensive hyperparameter tuning strategy

# Dataset
We use a large-scale Spotify dataset from Kaggle: https://www.kaggle.com/datasets/serkantysz/550k-spotify-songs-audio-lyrics-and-genres
- ~550K songs (subset of 25K used)
- Audio features (danceability, energy, tempo, etc.)
- Lyrics
- Genre labels

# Methodology
- Feature Processing
  - Audio features → standardized
  - Lyrics → embedded using Sentence-BERT
  - Genres → one-hot encoded
- Representation Learning
  - Basic VAE → dense latent vectors
  - ConvVAE → captures temporal/audio structure
  - CVAE → conditioned on genre + β-disentanglement
- Clustering Methods
  - K-Means
  - Agglomerative Clustering
  - DBSCAN
- Evaluation Metrics
  - Silhouette Score
  - Calinski-Harabasz Index
  - Davies-Bouldin Index
  - Adjusted Rand Index (ARI)
  - Normalized Mutual Information (NMI)
  - Cluster Purity

# Results Summary
The progressive approach shows consistent improvement:
- Basic VAE →	Better than PCA baseline
- ConvVAE →	Improved feature richness
- CVAE (β-VAE) → Best clustering performance

Final CVAE Performance:
- Silhouette Score: 0.3421
- NMI: 0.1603
- ARI: 0.1647
- Purity: 0.4173
It demonstrates stronger cluster separability and interpretability

# Visualization
We analyze latent spaces using:
- t-SNE
- UMAP

These show:
- More compact clusters
- Better genre separation
- Improved structure vs baseline

# Tech Stack
Python
PyTorch
Scikit-learn
SentenceTransformers (BERT)
Matplotlib / Seaborn

# Key Insight
Disentanglement + multimodal fusion is crucial for learning meaningful music representations.

The β-VAE + conditioning significantly improves clustering quality compared to:
- PCA
- Standard Autoencoders
- Spectral Clustering

# Future Work
- Better multilingual lyric embeddings
- Real-time recommendation systems
- Integration with audio waveform models

# Authors
Sabrina Jahan
Moin Mostakim
BRAC University, Bangladesh

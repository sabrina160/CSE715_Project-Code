# Multi-modal Beta-VAEs for Uncovering the Latent Structure of Music through Disentangled Representation and Interpretable Clustering
This project implements a progressive Variational Autoencoder (VAE) framework for uncovering latent structures in large-scale music data using audio features, lyrics, and genre information. The goal is to learn disentangled, interpretable representations that improve clustering quality in multi-modal music datasets.

# Overview
Traditional clustering methods struggle with music data due to:
- Nonlinear feature relationships
- Multi-modal complexity (audio + lyrics)
- Poor latent representation

To address this, this project proposes a 3-stage progressive learning pipeline:
- Basic VAE -> learns latent representation from audio
- Convolutional VAE (ConvVAE) -> captures local audio patterns
- Conditional / Beta-VAE (CVAE) -> enables disentangled multimodal learning

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

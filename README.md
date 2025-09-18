Hyperspectral & Thermal Anomaly Detection

This repository provides code and resources for spectral-spatial anomaly detection in hyperspectral and thermal (IR) remote sensing datasets. The goal is to detect manmade anomalies, suppress natural anomalies, and improve interpretability for applications such as environmental monitoring, precision agriculture, industrial activity tracking, wildfire detection, and disaster management.

⸻

🔍 Problem Overview
	•	Hyperspectral Images: Contain hundreds of contiguous spectral bands (400–2500 nm), where each pixel has a spectral signature enabling material identification.
	•	Thermal Images: Capture surface temperature variations, useful for detecting anomalies like fires, machinery operations, or industrial heat signatures.

Detecting anomalies in such datasets requires robust deep learning and generative AI approaches capable of handling high-dimensional data and fusing multiple sensor modalities.

⸻

🎯 Objectives
	•	Hyperspectral Anomaly Detection: Identify manmade anomalies from satellite hyperspectral images using deep learning.
	•	Spectral Characterization: Classify materials within anomalies using spectral analytics.
	•	Thermal Anomaly Detection: Detect anomalies caused by human activities in thermal images and video sequences.
	•	Thermal Characterization: Identify anomalies using material emissivity properties.
	•	Data Fusion: Combine hyperspectral and thermal modalities for more robust detection.

⸻

📂 Supported Datasets

The code supports open-source datasets for training and evaluation, including:
	•	Hyperspectral: PRISMA, EnMAP, Landsat (SWIR), and public benchmark datasets.
	•	Thermal: Landsat-8/9 thermal bands, UAV-based fire/IR datasets, and IR semantic segmentation datasets.

⸻

🛠️ Techniques Used

The repository integrates multiple techniques for anomaly detection:
	•	Deep Learning Models for spectral-spatial feature extraction.
	•	Generative AI for anomaly suppression and data-driven detection.
	•	Spectral Data Analytics for material identification.
	•	Thermal Emissivity Analysis for anomaly classification.
	•	Multimodal Fusion combining hyperspectral and thermal data.

⸻

🚀 Features
	•	Preprocessing pipeline for hyperspectral and thermal data.
	•	Anomaly detection using deep learning models.
	•	Outputs in GeoTIFF and PNG for geospatial visualization.
	•	Evaluation metrics: F1 Score, ROC-AUC, PR-AUC.
	•	Support for large-scale images (30 km x 30 km, 5–30 m resolution).

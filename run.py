import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from typing import Tuple, Dict, List, Optional
import warnings
import os
warnings.filterwarnings('ignore')

class SpectralSpatialAutoEncoder(nn.Module):
    """
    Spectral-Spatial Autoencoder for Hyperspectral Anomaly Detection
    Based on the research findings showing autoencoders perform well for HSI-AD
    """
    def __init__(self, spectral_dim: int, spatial_size: int = 5, latent_dim: int = 64):
        super(SpectralSpatialAutoEncoder, self).__init__()
        self.spectral_dim = spectral_dim
        self.spatial_size = spatial_size
        self.latent_dim = latent_dim

        # Spectral encoder
        self.spectral_encoder = nn.Sequential(
            nn.Linear(spectral_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, latent_dim)
        )

        # Spatial encoder (1D CNN for spectral-spatial features)
        self.spatial_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(latent_dim)
        )

        # Fusion layer
        self.fusion = nn.Linear(latent_dim * 2, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, spectral_dim),
            nn.Sigmoid()
        )

    def forward(self, x_spectral, x_spatial=None):
        # Spectral encoding
        spectral_encoded = self.spectral_encoder(x_spectral)

        if x_spatial is not None:
            # Spatial encoding
            spatial_encoded = self.spatial_encoder(x_spatial.unsqueeze(1))
            spatial_encoded = spatial_encoded.squeeze(-1)

            # Fusion
            fused = torch.cat([spectral_encoded, spatial_encoded], dim=1)
            latent = self.fusion(fused)
        else:
            latent = spectral_encoded

        # Decoding
        reconstructed = self.decoder(latent)
        return reconstructed, latent

class VariationalAutoEncoder(nn.Module):
    """
    Variational Autoencoder for better anomaly detection
    Based on research showing VAE effectiveness for HSI-AD
    """
    def __init__(self, input_dim: int, latent_dim: int = 64):
        super(VariationalAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.mu_layer = nn.Linear(128, latent_dim)
        self.logvar_layer = nn.Linear(128, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

class HyperspectralAnomalyDetector:
    """
    Complete Hyperspectral Anomaly Detection Pipeline
    Handles .mat files and provides visualization
    """
    def __init__(self, model_type='autoencoder', spectral_dim=None, spatial_size=5):
        self.model_type = model_type
        self.spectral_dim = spectral_dim
        self.spatial_size = spatial_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False

        print(f"Initialized detector on {self.device}")

    def load_mat_file(self, file_path: str, data_key: Optional[str] = None,
                     gt_key: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load hyperspectral data from .mat file

        Args:
            file_path: Path to .mat file
            data_key: Key for hyperspectral data in .mat file (if None, auto-detect)
            gt_key: Key for ground truth data (optional)

        Returns:
            Tuple of (hyperspectral_data, ground_truth)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        print(f"Loading .mat file: {file_path}")
        mat_data = sio.loadmat(file_path)

        # Remove metadata keys
        data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        print(f"Available keys in .mat file: {data_keys}")

        # Auto-detect data key if not provided
        if data_key is None:
            # Common naming conventions for hyperspectral data
            common_keys = ['data', 'hsi', 'img', 'image', 'cube', 'hyperspectral']
            for key in common_keys:
                if key in mat_data:
                    data_key = key
                    break

            if data_key is None:
                # Use the key with largest 3D array
                for key in data_keys:
                    if isinstance(mat_data[key], np.ndarray) and mat_data[key].ndim == 3:
                        data_key = key
                        break

        if data_key is None or data_key not in mat_data:
            raise ValueError(f"Could not find hyperspectral data. Available keys: {data_keys}")

        hsi_data = mat_data[data_key]
        print(f"Loaded HSI data with key '{data_key}': {hsi_data.shape}")

        # Load ground truth if available
        ground_truth = None
        if gt_key and gt_key in mat_data:
            ground_truth = mat_data[gt_key]
            print(f"Loaded ground truth with key '{gt_key}': {ground_truth.shape}")
        elif 'gt' in mat_data:
            ground_truth = mat_data['gt']
            print(f"Auto-detected ground truth: {ground_truth.shape}")

        # Ensure proper data format (H x W x C)
        if hsi_data.ndim != 3:
            raise ValueError(f"Expected 3D hyperspectral data, got shape: {hsi_data.shape}")

        # Update spectral dimension
        if self.spectral_dim is None:
            self.spectral_dim = hsi_data.shape[-1]
            print(f"Auto-detected spectral dimension: {self.spectral_dim}")

        return hsi_data, ground_truth

    def _initialize_model(self):
        """Initialize the model with correct spectral dimension"""
        if self.spectral_dim is None:
            raise ValueError("Spectral dimension not set. Load data first.")

        if self.model_type == 'autoencoder':
            self.model = SpectralSpatialAutoEncoder(self.spectral_dim, self.spatial_size)
        elif self.model_type == 'vae':
            self.model = VariationalAutoEncoder(self.spectral_dim)
        else:
            raise ValueError("model_type must be 'autoencoder' or 'vae'")

        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        print(f"Initialized {self.model_type} model with {self.spectral_dim} spectral bands")

    def preprocess_data(self, data: np.ndarray, fit_scaler: bool = False) -> torch.Tensor:
        """Preprocess hyperspectral data"""
        # Normalize data
        original_shape = data.shape
        data_2d = data.reshape(-1, data.shape[-1])

        if fit_scaler:
            data_normalized = self.scaler.fit_transform(data_2d)
        else:
            data_normalized = self.scaler.transform(data_2d)

        data_normalized = data_normalized.reshape(original_shape)

        # Convert to tensor
        return torch.FloatTensor(data_normalized).to(self.device)

    def vae_loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        """VAE loss function with KL divergence"""
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + beta * KLD

    def spectral_angle_loss(self, x, x_recon):
        """Spectral Angle Mapper loss for better spectral preservation"""
        dot_product = torch.sum(x * x_recon, dim=1)
        norm_x = torch.norm(x, dim=1)
        norm_recon = torch.norm(x_recon, dim=1)
        cos_angle = dot_product / (norm_x * norm_recon + 1e-8)
        spectral_angle = torch.acos(torch.clamp(cos_angle, -1, 1))
        return torch.mean(spectral_angle)

    def train(self, hsi_data: np.ndarray, epochs: int = 100, batch_size: int = 256):
        """Train the anomaly detection model"""
        if self.model is None:
            self._initialize_model()

        print(f"Training {self.model_type} on {self.device}")
        print(f"Data shape: {hsi_data.shape}")

        # Preprocess data
        train_data_flat = hsi_data.reshape(-1, hsi_data.shape[-1])
        train_tensor = self.preprocess_data(train_data_flat, fit_scaler=True)

        self.model.train()
        train_losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            # Mini-batch training
            for i in range(0, len(train_tensor), batch_size):
                batch = train_tensor[i:i+batch_size]

                self.optimizer.zero_grad()

                if self.model_type == 'autoencoder':
                    reconstructed, _ = self.model(batch)
                    # Combined loss: MSE + Spectral Angle
                    mse_loss = F.mse_loss(reconstructed, batch)
                    sa_loss = self.spectral_angle_loss(batch, reconstructed)
                    loss = mse_loss + 0.1 * sa_loss
                else:  # VAE
                    recon_batch, mu, logvar = self.model(batch)
                    loss = self.vae_loss_function(recon_batch, batch, mu, logvar)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)

            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')

        self.is_trained = True
        print("Training completed!")
        return train_losses

    def detect_anomalies(self, hsi_data: np.ndarray, return_reconstruction: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies in hyperspectral data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before detection. Call train() first.")

        self.model.eval()

        original_shape = hsi_data.shape
        test_data_flat = hsi_data.reshape(-1, hsi_data.shape[-1])
        test_tensor = self.preprocess_data(test_data_flat, fit_scaler=False)

        anomaly_scores = []
        reconstruction_errors = []
        reconstructed_data = []

        with torch.no_grad():
            batch_size = 1000
            for i in range(0, len(test_tensor), batch_size):
                batch = test_tensor[i:i+batch_size]

                if self.model_type == 'autoencoder':
                    reconstructed, latent = self.model(batch)
                    # Reconstruction error as anomaly score
                    recon_error = torch.mean((batch - reconstructed) ** 2, dim=1)
                    # Additional spectral angle error
                    sa_error = torch.mean(torch.abs(batch - reconstructed), dim=1)
                    combined_error = recon_error + 0.1 * sa_error
                else:  # VAE
                    recon_batch, mu, logvar = self.model(batch)
                    # Reconstruction probability as anomaly score
                    recon_error = torch.mean((batch - recon_batch) ** 2, dim=1)
                    combined_error = recon_error
                    reconstructed = recon_batch

                anomaly_scores.extend(combined_error.cpu().numpy())
                reconstruction_errors.extend(recon_error.cpu().numpy())

                if return_reconstruction:
                    reconstructed_data.extend(reconstructed.cpu().numpy())

        anomaly_scores = np.array(anomaly_scores)
        reconstruction_errors = np.array(reconstruction_errors)

        # Reshape back to original spatial dimensions
        anomaly_map = anomaly_scores.reshape(original_shape[:2])
        recon_error_map = reconstruction_errors.reshape(original_shape[:2])

        if return_reconstruction:
            reconstructed_cube = np.array(reconstructed_data).reshape(original_shape)
            return anomaly_map, recon_error_map, reconstructed_cube

        return anomaly_map, recon_error_map

    def visualize_results(self, hsi_data: np.ndarray, anomaly_map: np.ndarray,
                         ground_truth: Optional[np.ndarray] = None,
                         threshold_percentile: float = 95,
                         save_path: Optional[str] = None):
        """
        Visualize anomaly detection results

        Args:
            hsi_data: Original hyperspectral data
            anomaly_map: Detected anomaly scores
            ground_truth: Optional ground truth for comparison
            threshold_percentile: Percentile for anomaly threshold
            save_path: Optional path to save the visualization
        """
        # Calculate threshold
        threshold = np.percentile(anomaly_map, threshold_percentile)
        binary_anomalies = (anomaly_map > threshold).astype(int)

        # Create RGB visualization of HSI data
        if hsi_data.shape[-1] >= 3:
            # Use bands as RGB (approximate)
            rgb_bands = [min(int(hsi_data.shape[-1] * 0.7), hsi_data.shape[-1]-1),  # Red
                        min(int(hsi_data.shape[-1] * 0.5), hsi_data.shape[-1]-1),  # Green
                        min(int(hsi_data.shape[-1] * 0.3), hsi_data.shape[-1]-1)]  # Blue
            rgb_image = hsi_data[:, :, rgb_bands]
            # Normalize for display
            rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
        else:
            # Use first band if insufficient bands
            rgb_image = np.stack([hsi_data[:, :, 0]] * 3, axis=-1)
            rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())

        # Set up the plot
        if ground_truth is not None:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Original RGB representation
        axes[0, 0].imshow(rgb_image)
        axes[0, 0].set_title('HSI RGB Representation', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        # Anomaly score map
        im1 = axes[0, 1].imshow(anomaly_map, cmap='hot', interpolation='nearest')
        axes[0, 1].set_title(f'Anomaly Scores ({self.model_type.upper()})', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        # Binary anomaly detection
        colors = ['blue', 'red']
        cmap = ListedColormap(colors)
        im2 = axes[1, 0].imshow(binary_anomalies, cmap=cmap, interpolation='nearest')
        axes[1, 0].set_title(f'Detected Anomalies (>{threshold_percentile}th percentile)',
                            fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')

        # Create legend for binary map
        legend_elements = [mpatches.Patch(color='blue', label='Background'),
                          mpatches.Patch(color='red', label='Anomaly')]
        axes[1, 0].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))

        # Overlay on RGB
        overlay = rgb_image.copy()
        anomaly_mask = binary_anomalies == 1
        overlay[anomaly_mask] = [1, 0, 0]  # Red overlay for anomalies

        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Anomalies Overlaid on RGB', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')

        # Add ground truth comparison if available
        if ground_truth is not None:
            # Ground truth
            im3 = axes[0, 2].imshow(ground_truth, cmap=cmap, interpolation='nearest')
            axes[0, 2].set_title('Ground Truth', fontsize=14, fontweight='bold')
            axes[0, 2].axis('off')
            axes[0, 2].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))

            # Performance comparison
            tp = np.sum((binary_anomalies == 1) & (ground_truth == 1))
            fp = np.sum((binary_anomalies == 1) & (ground_truth == 0))
            fn = np.sum((binary_anomalies == 0) & (ground_truth == 1))
            tn = np.sum((binary_anomalies == 0) & (ground_truth == 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Create confusion-like visualization
            comparison = np.zeros_like(ground_truth, dtype=int)
            comparison[(binary_anomalies == 1) & (ground_truth == 1)] = 3  # TP - Green
            comparison[(binary_anomalies == 1) & (ground_truth == 0)] = 2  # FP - Orange
            comparison[(binary_anomalies == 0) & (ground_truth == 1)] = 1  # FN - Yellow
            # TN remains 0 - Blue

            comp_colors = ['blue', 'yellow', 'orange', 'green']
            comp_cmap = ListedColormap(comp_colors)
            axes[1, 2].imshow(comparison, cmap=comp_cmap, interpolation='nearest')
            axes[1, 2].set_title(f'Performance Map\nF1: {f1:.3f}, Prec: {precision:.3f}, Rec: {recall:.3f}',
                               fontsize=12, fontweight='bold')
            axes[1, 2].axis('off')

            # Performance legend
            perf_legend = [mpatches.Patch(color='blue', label=f'TN ({tn})'),
                          mpatches.Patch(color='yellow', label=f'FN ({fn})'),
                          mpatches.Patch(color='orange', label=f'FP ({fp})'),
                          mpatches.Patch(color='green', label=f'TP ({tp})')]
            axes[1, 2].legend(handles=perf_legend, loc='upper right', bbox_to_anchor=(1, 1), fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")

        plt.show()

        # Print summary statistics
        print(f"\nANOMALY DETECTION SUMMARY")
        print(f"="*40)
        print(f"Model type: {self.model_type.upper()}")
        print(f"Image shape: {hsi_data.shape}")
        print(f"Threshold (percentile {threshold_percentile}): {threshold:.6f}")
        print(f"Detected anomaly pixels: {np.sum(binary_anomalies)}")
        print(f"Anomaly ratio: {np.mean(binary_anomalies):.4f}")

        if ground_truth is not None:
            print(f"Ground truth anomalies: {np.sum(ground_truth)}")
            print(f"True anomaly ratio: {np.mean(ground_truth):.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")

    def evaluate(self, hsi_data: np.ndarray, ground_truth: np.ndarray) -> Dict:
        """Evaluate the model performance with ground truth"""
        anomaly_map, _ = self.detect_anomalies(hsi_data)

        # Flatten for evaluation
        scores_flat = anomaly_map.flatten()
        gt_flat = ground_truth.flatten()

        # Calculate metrics
        auc_roc = roc_auc_score(gt_flat, scores_flat)

        # Calculate PR-AUC
        precision, recall, _ = precision_recall_curve(gt_flat, scores_flat)
        auc_pr = auc(recall, precision)

        # Calculate F1 score with optimal threshold
        thresholds = np.percentile(scores_flat, np.arange(90, 99.9, 0.1))
        best_f1 = 0
        best_threshold = 0

        for threshold in thresholds:
            predictions = (scores_flat > threshold).astype(int)
            f1 = f1_score(gt_flat, predictions)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        results = {
            'AUC-ROC': auc_roc,
            'AUC-PR': auc_pr,
            'F1-Score': best_f1,
            'Best_Threshold': best_threshold
        }

        return results

# Example usage function
def process_mat_file(file_path: str,
                    model_type: str = 'autoencoder',
                    data_key: Optional[str] = None,
                    gt_key: Optional[str] = None,
                    epochs: int = 100,
                    threshold_percentile: float = 95,
                    save_visualization: bool = True):
    """
    Complete pipeline to process a .mat file and detect anomalies

    Args:
        file_path: Path to .mat file
        model_type: 'autoencoder' or 'vae'
        data_key: Key for HSI data in .mat file
        gt_key: Key for ground truth (optional)
        epochs: Training epochs
        threshold_percentile: Threshold for anomaly detection
        save_visualization: Whether to save visualization
    """
    print("HYPERSPECTRAL ANOMALY DETECTION")
    print("="*50)

    # Initialize detector
    detector = HyperspectralAnomalyDetector(model_type=model_type)

    # Load data
    hsi_data, ground_truth = detector.load_mat_file(file_path, data_key, gt_key)

    # Train model
    print(f"\nTraining {model_type} model...")
    train_losses = detector.train(hsi_data, epochs=epochs)

    # Detect anomalies
    print("\nDetecting anomalies...")
    anomaly_map, recon_error_map = detector.detect_anomalies(hsi_data)

    # Visualize results
    save_path = f"{os.path.splitext(file_path)[0]}_anomaly_detection.png" if save_visualization else None
    detector.visualize_results(hsi_data, anomaly_map, ground_truth,
                              threshold_percentile, save_path)

    # Evaluate if ground truth available
    if ground_truth is not None:
        results = detector.evaluate(hsi_data, ground_truth)
        print(f"\nPERFORMANCE METRICS")
        print(f"="*30)
        print(f"AUC-ROC: {results['AUC-ROC']:.4f}")
        print(f"AUC-PR: {results['AUC-PR']:.4f}")
        print(f"F1-Score: {results['F1-Score']:.4f}")

    return detector, anomaly_map

# Example usage
if __name__ == "__main__":
    # Example: Process your .mat file
    # Replace with your actual .mat file path

    mat_file_path = "/Users/chinmayk/Desktop/Python day project/PRS_L2D_STD_20201214060713_20201214060717_0001.mat"

    print("")
    print("="*60)
    print("Usage example:")
    print("detector, anomaly_map = process_mat_file('your_file.mat')")
    print("\nFor custom keys:")
    print("detector, anomaly_map = process_mat_file('your_file.mat', data_key='hsi_data', gt_key='ground_truth')")
    detector, anomaly_map = process_mat_file(mat_file_path,
                                            model_type='autoencoder',
                                             epochs=100,
                                             threshold_percentile=95)
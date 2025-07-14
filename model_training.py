import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import streamlit as st

class SolarProjectDataset(Dataset):
    """Dataset class for solar project training data"""
    
    def __init__(self, features, targets, transform=None):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.transform = transform
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = {
            'features': self.features[idx],
            'targets': self.targets[idx]
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

class AdvancedSolarLayoutNN(nn.Module):
    """Enhanced neural network for solar layout optimization"""
    
    def __init__(self, input_dim=25, hidden_dims=[256, 128, 64, 32], dropout_rate=0.2):
        super(AdvancedSolarLayoutNN, self).__init__()
        
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        
        # Feature extraction layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Multi-head output for different optimization targets
        self.capacity_head = nn.Sequential(
            nn.Linear(prev_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),  # Predicted capacity (MW)
            nn.Sigmoid()
        )
        
        self.efficiency_head = nn.Sequential(
            nn.Linear(prev_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),  # Efficiency score (0-1)
            nn.Sigmoid()
        )
        
        self.layout_params_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # [optimal_gcr, optimal_pitch, optimal_angle, spacing_factor]
        )
        
        self.cost_head = nn.Sequential(
            nn.Linear(prev_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),  # Cost per MW
            nn.ReLU()
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        
        capacity = self.capacity_head(features)
        efficiency = self.efficiency_head(features)
        layout_params = self.layout_params_head(features)
        cost = self.cost_head(features)
        
        return {
            'capacity': capacity,
            'efficiency': efficiency,
            'layout_params': layout_params,
            'cost': cost
        }

class SolarAITrainer:
    """Training class for solar layout optimization models"""
    
    def __init__(self, model_config=None):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.training_history = []
        self.best_model_state = None
        self.best_loss = float('inf')
        
        self.config = model_config or {
            'input_dim': 25,
            'hidden_dims': [256, 128, 64, 32],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'early_stopping_patience': 15
        }
    
    def create_synthetic_data(self, n_samples=5000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data for demonstration"""
        np.random.seed(42)
        
        # Generate realistic feature distributions
        X = np.random.rand(n_samples, 25)
        
        # Realistic feature scaling
        X[:, 0] = X[:, 0] * 2000 + 100      # total_acres (100-2100)
        X[:, 1] = X[:, 1] * X[:, 0] * 0.9   # buildable_acres (90% of total max)
        X[:, 2] = (X[:, 1] / X[:, 0]) * 100 # buildable_pct
        X[:, 3] = X[:, 3] * 25              # slope_avg (0-25%)
        X[:, 4] = X[:, 4] * 10              # slope_variance
        X[:, 5] = X[:, 5] * 0.3             # wetlands_pct (0-30%)
        X[:, 6] = X[:, 6] * 0.2             # floodplain_pct (0-20%)
        X[:, 7] = X[:, 7] * 0.15            # bedrock_pct (0-15%)
        X[:, 8] = X[:, 8] * 20 + 5          # setback_m (5-25)
        X[:, 9] = X[:, 9] * 100 + 50        # tracker_length_m (50-150)
        X[:, 10] = X[:, 10] * 2 + 2         # tracker_width_m (2-4)
        X[:, 11] = X[:, 11] * 0.6 + 0.2     # gcr (0.2-0.8)
        X[:, 12] = X[:, 12]                 # orientation (0-1)
        X[:, 13] = X[:, 13] * 8 + 4         # irradiance (4-12 kWh/m²/day)
        X[:, 14] = X[:, 14] * 30 + 5        # temperature (5-35°C)
        X[:, 15] = X[:, 15] * 2000          # elevation (0-2000m)
        
        # Add derived features
        X[:, 16] = X[:, 0] / (X[:, 1] * 0.5)     # acres_per_mw
        X[:, 17] = X[:, 5] + X[:, 6] + X[:, 7]   # constraint_density
        X[:, 18] = X[:, 3] * X[:, 4]             # terrain_complexity
        X[:, 19] = X[:, 13] / X[:, 14]           # weather_score
        
        # Additional synthetic features
        X[:, 20:25] = np.random.rand(n_samples, 5)
        
        # Generate realistic targets based on features
        capacity = (X[:, 1] * 0.5 * X[:, 11] * (1 - X[:, 17] * 0.3) + 
                   np.random.normal(0, 10, n_samples))  # capacity_mw
        efficiency = (0.8 + X[:, 13] * 0.05 - X[:, 3] * 0.01 - X[:, 17] * 0.2 + 
                     np.random.normal(0, 0.05, n_samples))  # efficiency_score
        
        # Layout parameters (these would be optimized)
        opt_gcr = np.clip(X[:, 11] + np.random.normal(0, 0.1, n_samples), 0.2, 0.8)
        opt_length = np.clip(X[:, 9] + np.random.normal(0, 10, n_samples), 50, 150)
        opt_orientation = np.random.choice([0, 1], n_samples)
        opt_setback = np.clip(X[:, 8] + np.random.normal(0, 2, n_samples), 5, 25)
        
        # Cost (inversely related to efficiency, scale effects)
        cost = (2000000 - efficiency * 500000 + X[:, 18] * 100000 + 
               np.random.normal(0, 100000, n_samples))  # cost_per_mw
        
        y = np.column_stack([
            capacity, efficiency, opt_gcr, opt_length, 
            opt_orientation, opt_setback, cost
        ])
        
        return X, y
    
    def train_model(self, X: np.ndarray, y: np.ndarray, validation_split=0.2):
        """Train the neural network model"""
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # Create datasets
        train_dataset = SolarProjectDataset(X_train, y_train)
        val_dataset = SolarProjectDataset(X_val, y_val)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False
        )
        
        # Initialize model
        self.model = AdvancedSolarLayoutNN(
            input_dim=X.shape[1],
            hidden_dims=self.config['hidden_dims'],
            dropout_rate=self.config['dropout_rate']
        )
        
        # Loss functions for multi-task learning
        criterion_mse = nn.MSELoss()
        criterion_mae = nn.L1Loss()
        
        # Optimizer
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Training loop
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            # Training phase
            self.model.train()
            train_losses = []
            
            for batch in train_loader:
                features = batch['features']
                targets = batch['targets']
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(features)
                
                # Calculate losses for each head
                capacity_loss = criterion_mse(outputs['capacity'].squeeze(), targets[:, 0])
                efficiency_loss = criterion_mse(outputs['efficiency'].squeeze(), targets[:, 1])
                layout_loss = criterion_mse(outputs['layout_params'], targets[:, 2:6])
                cost_loss = criterion_mae(outputs['cost'].squeeze(), targets[:, 6])
                
                # Combined loss with weights
                total_loss = (0.3 * capacity_loss + 0.3 * efficiency_loss + 
                             0.3 * layout_loss + 0.1 * cost_loss)
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_losses.append(total_loss.item())
            
            # Validation phase
            self.model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features']
                    targets = batch['targets']
                    
                    outputs = self.model(features)
                    
                    capacity_loss = criterion_mse(outputs['capacity'].squeeze(), targets[:, 0])
                    efficiency_loss = criterion_mse(outputs['efficiency'].squeeze(), targets[:, 1])
                    layout_loss = criterion_mse(outputs['layout_params'], targets[:, 2:6])
                    cost_loss = criterion_mae(outputs['cost'].squeeze(), targets[:, 6])
                    
                    total_loss = (0.3 * capacity_loss + 0.3 * efficiency_loss + 
                                 0.3 * layout_loss + 0.1 * cost_loss)
                    
                    val_losses.append(total_loss.item())
            
            # Calculate average losses
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Save training history
            self.training_history.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            # Early stopping and best model saving
            if avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, "
                      f"Val Loss = {avg_val_loss:.4f}")
            
            # Early stopping
            if patience_counter >= self.config['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        print("Training completed!")
        return self.training_history
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model performance"""
        if not self.model:
            raise ValueError("Model not trained yet!")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test)
            outputs = self.model(X_tensor)
            
            # Extract predictions
            capacity_pred = outputs['capacity'].squeeze().numpy()
            efficiency_pred = outputs['efficiency'].squeeze().numpy()
            layout_pred = outputs['layout_params'].numpy()
            cost_pred = outputs['cost'].squeeze().numpy()
            
            # Calculate metrics
            metrics = {
                'capacity_r2': r2_score(y_test[:, 0], capacity_pred),
                'capacity_rmse': np.sqrt(mean_squared_error(y_test[:, 0], capacity_pred)),
                'efficiency_r2': r2_score(y_test[:, 1], efficiency_pred),
                'efficiency_rmse': np.sqrt(mean_squared_error(y_test[:, 1], efficiency_pred)),
                'layout_r2': r2_score(y_test[:, 2:6], layout_pred),
                'layout_rmse': np.sqrt(mean_squared_error(y_test[:, 2:6], layout_pred)),
                'cost_r2': r2_score(y_test[:, 6], cost_pred),
                'cost_rmse': np.sqrt(mean_squared_error(y_test[:, 6], cost_pred))
            }
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save trained model and preprocessing components"""
        if not self.model:
            raise ValueError("No model to save!")
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': self.config,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'training_history': self.training_history,
            'best_loss': self.best_loss
        }
        
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model and preprocessing components"""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        self.config = checkpoint['model_config']
        self.model = AdvancedSolarLayoutNN(
            input_dim=self.config['input_dim'],
            hidden_dims=self.config['hidden_dims'],
            dropout_rate=self.config['dropout_rate']
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.label_encoders = checkpoint['label_encoders']
        self.training_history = checkpoint['training_history']
        self.best_loss = checkpoint['best_loss']
        
        print(f"Model loaded from {filepath}")
    
    def plot_training_history(self):
        """Plot training history"""
        if not self.training_history:
            print("No training history available")
            return
        
        df_history = pd.DataFrame(self.training_history)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(df_history['epoch'], df_history['train_loss'], label='Training Loss')
        ax1.plot(df_history['epoch'], df_history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training History')
        ax1.legend()
        ax1.grid(True)
        
        # Learning rate plot
        ax2.plot(df_history['epoch'], df_history['learning_rate'])
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')
        ax2.grid(True)
        
        plt.tight_layout()
        return fig

def create_training_interface():
    """Streamlit interface for model training"""
    st.title("🤖 AI Model Training Interface")
    
    st.markdown("""
    Train custom neural networks on your historical solar project data to improve optimization accuracy.
    """)
    
    # Training configuration
    st.sidebar.header("🔧 Training Configuration")
    
    epochs = st.sidebar.slider("Epochs", 10, 500, 100)
    batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    learning_rate = st.sidebar.number_input("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
    
    # Data source selection
    st.header("📊 Training Data")
    data_source = st.radio(
        "Choose data source:",
        ["Use synthetic data (demo)", "Upload historical project data (CSV)"]
    )
    
    trainer = SolarAITrainer({
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    })
    
    if data_source == "Use synthetic data (demo)":
        if st.button("🚀 Start Training with Synthetic Data"):
            with st.spinner("Generating synthetic data and training model..."):
                # Generate synthetic data
                X, y = trainer.create_synthetic_data(n_samples=2000)
                
                # Train model
                history = trainer.train_model(X, y)
                
                # Display results
                st.success("✅ Training completed!")
                
                # Plot training history
                fig = trainer.plot_training_history()
                st.pyplot(fig)
                
                # Evaluate on test set
                X_test, y_test = trainer.create_synthetic_data(n_samples=500)
                X_test = trainer.scaler.transform(X_test)
                metrics = trainer.evaluate_model(X_test, y_test)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Capacity R²", f"{metrics['capacity_r2']:.3f}")
                with col2:
                    st.metric("Efficiency R²", f"{metrics['efficiency_r2']:.3f}")
                with col3:
                    st.metric("Layout R²", f"{metrics['layout_r2']:.3f}")
                with col4:
                    st.metric("Cost R²", f"{metrics['cost_r2']:.3f}")
                
                # Save model option
                if st.button("💾 Save Trained Model"):
                    model_path = "models/trained_solar_ai_model.pth"
                    os.makedirs("models", exist_ok=True)
                    trainer.save_model(model_path)
                    st.success(f"Model saved to {model_path}")

if __name__ == "__main__":
    # Example usage
    trainer = SolarAITrainer()
    
    print("Generating synthetic training data...")
    X, y = trainer.create_synthetic_data(n_samples=1000)
    
    print("Training model...")
    history = trainer.train_model(X, y)
    
    print("Evaluating model...")
    X_test, y_test = trainer.create_synthetic_data(n_samples=200)
    X_test = trainer.scaler.transform(X_test)
    metrics = trainer.evaluate_model(X_test, y_test)
    
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    trainer.save_model("models/solar_ai_model.pth")
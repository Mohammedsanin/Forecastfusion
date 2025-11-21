# Multi-Model Time Series Forecasting Web App

## Overview

This is a Streamlit-based web application for time series forecasting that enables users to train and compare multiple deep learning models (ANN, RNN, and LSTM) on their time series data. The application provides an interactive interface for uploading CSV data, configuring model parameters, training multiple models simultaneously, and visualizing predictions with performance metrics. It supports both single-step and multi-step forecasting capabilities with downloadable prediction results.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Framework**: Streamlit with custom CSS styling
- **Layout Pattern**: Wide layout with sidebar for controls and main area for visualizations
- **Component Structure**: Single-page application with sequential workflow (upload → configure → train → visualize)
- **Visualization Engine**: Plotly for interactive charts and graphs
- **State Management**: Streamlit's session state for maintaining user inputs and model results

**Design Rationale**: Streamlit was chosen for rapid prototyping and deployment, eliminating the need for separate frontend/backend development. Plotly provides rich, interactive visualizations essential for time series analysis.

### Backend Architecture

**Machine Learning Pipeline**:
- **Data Preprocessing**: MinMaxScaler for feature normalization (0-1 range)
- **Sequence Generation**: Sliding window approach for creating input-output pairs
- **Model Training**: TensorFlow/Keras for neural network implementations
- **Evaluation**: RMSE (Root Mean Squared Error) for performance comparison

**Model Architectures**:

1. **ANN (Artificial Neural Network)**
   - Feedforward architecture with Dense layers (64→32→1 neurons)
   - Use case: Simple patterns and short-term predictions
   - Trade-off: Fast training but limited temporal dependency capture

2. **RNN (Recurrent Neural Network)**
   - Stacked SimpleRNN layers (50→25 units) with sequence processing
   - Use case: Basic sequential dependencies
   - Trade-off: Handles sequences but prone to vanishing gradient issues

3. **LSTM (Long Short-Term Memory)**
   - Stacked LSTM layers (64→32 units) with Dense output layers
   - Use case: Complex patterns and long-term dependencies
   - Trade-off: Best performance but higher computational cost

**Design Decisions**:
- Multiple model support allows users to compare approaches and select the best performer
- Configurable hyperparameters (sequence length, epochs, batch size) provide flexibility without overwhelming users
- Reproducible results via fixed random seeds (seed=42)

### Data Processing Strategy

**Input Requirements**: CSV file with single numeric column representing time series data
- **Scaling**: MinMaxScaler chosen to maintain gradient stability during training
- **Train-Test Split**: Typically 80-20 split for evaluation
- **Sequence Creation**: Configurable window size (10-100 steps) for pattern recognition

**Multi-Step Forecasting**:
- Recursive prediction approach: using model outputs as future inputs
- Configurable forecast horizon (1-50 steps)
- Enables scenario planning and longer-term predictions

### Performance & Optimization

**Computational Considerations**:
- TensorFlow backend with GPU support (if available)
- Adam optimizer for efficient gradient descent
- Dropout layers (implied in architecture) for regularization
- Batch processing for memory efficiency

**Evaluation Metrics**:
- RMSE as primary metric for model comparison
- Visual inspection through actual vs. predicted plots
- Comparative analysis across all trained models

## External Dependencies

### Core Libraries

**Data Processing & Scientific Computing**:
- `numpy`: Array operations and numerical computations (seed: 42)
- `pandas`: CSV data loading and DataFrame manipulation
- `scikit-learn`: Data preprocessing (MinMaxScaler) and metrics (mean_squared_error)

**Deep Learning Framework**:
- `tensorflow/keras`: Neural network model building and training
  - Sequential API for model construction
  - Layer types: Dense, SimpleRNN, LSTM, Dropout
  - Adam optimizer for training

**Visualization**:
- `plotly`: Interactive plotting library
  - `plotly.graph_objects`: Custom chart creation
  - `plotly.express`: Quick visualization templates
  - `plotly.subplots`: Multi-model comparison plots

**Web Framework**:
- `streamlit`: Web application framework
  - File upload widgets
  - Interactive controls (sliders, number inputs, checkboxes)
  - Download functionality for predictions
  - Custom CSS styling support

### Data Requirements

**Input Format**: CSV file with time series data
- Single numeric column expected
- Sequential ordering implied (time-indexed)
- No external database required (file-based input)

**Output Format**: 
- Interactive Plotly visualizations
- Downloadable CSV predictions
- RMSE comparison table

### Development & Deployment

**Platform**: Replit-compatible
- No external APIs or authentication required
- Self-contained Python environment
- Browser-based execution via Streamlit server

**Note**: Application is currently stateless (no persistent storage). All data and models exist only during session runtime.

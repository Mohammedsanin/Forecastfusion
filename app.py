import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Dropout
from tensorflow.keras.optimizers import Adam
from statsmodels.tsa.seasonal import seasonal_decompose
import io
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Page configuration
st.set_page_config(
    page_title="Multi-Model Time Series Forecasting",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00AA00;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def create_demo_data():
    """Create demo time series data for testing purposes"""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
    
    # Create a synthetic time series with trend, seasonality, and noise
    trend = np.linspace(100, 200, 500)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(500) / 365.25)
    noise = np.random.normal(0, 5, 500)
    values = trend + seasonal + noise
    
    df = pd.DataFrame({
        'Date': dates,
        'Value': values
    })
    return df

def preprocess_data(data, time_steps, target_column):
    """
    Preprocess data by creating sliding windows for time series forecasting
    
    Args:
        data: DataFrame with time series data
        time_steps: Number of previous time steps to use as input
        target_column: Name of the target column
    
    Returns:
        X, y: Input sequences and targets
        scaler: Fitted MinMaxScaler
        original_data: Original data for plotting
    """
    # Extract the target column
    values = data[target_column].values.reshape(-1, 1)
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values)
    
    # Create sequences
    X, y = [], []
    for i in range(time_steps, len(scaled_values)):
        X.append(scaled_values[i-time_steps:i, 0])
        y.append(scaled_values[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    return X, y, scaler, values

def build_ann_model(time_steps):
    """Build Artificial Neural Network model"""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(time_steps,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def build_rnn_model(time_steps):
    """Build Vanilla RNN model"""
    model = Sequential([
        SimpleRNN(50, return_sequences=True, input_shape=(time_steps, 1)),
        Dropout(0.2),
        SimpleRNN(25),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def build_lstm_model(time_steps):
    """Build LSTM model"""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(time_steps, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def build_gru_model(time_steps):
    """Build GRU model"""
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=(time_steps, 1)),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def train_model(model, X_train, y_train, epochs, batch_size, validation_split=0.2):
    """Train a model with given parameters"""
    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=0
    )
    
    return model, history

def predict_future_steps(model, last_sequence, scaler, n_steps, model_type):
    """Predict multiple future steps"""
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(n_steps):
        if model_type == 'ANN':
            # For ANN, use the sequence as is
            pred_input = current_sequence.reshape(1, -1)
        else:
            # For RNN/LSTM/GRU, reshape to (1, time_steps, 1)
            pred_input = current_sequence.reshape(1, len(current_sequence), 1)
        
        next_pred = model.predict(pred_input, verbose=0)[0, 0]
        predictions.append(next_pred)
        
        # Update sequence for next prediction
        current_sequence = np.append(current_sequence[1:], next_pred)
    
    # Inverse transform predictions
    predictions_array = np.array(predictions).reshape(-1, 1)
    predictions_inverse = scaler.inverse_transform(predictions_array)
    
    return predictions_inverse.flatten()

def create_prediction_plot(y_true, y_pred, model_name, future_predictions=None):
    """Create interactive plot for actual vs predicted values"""
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=[f'{model_name} - Actual vs Predicted']
    )
    
    # Add actual values
    fig.add_trace(
        go.Scatter(
            y=y_true,
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2)
        )
    )
    
    # Add predictions
    fig.add_trace(
        go.Scatter(
            y=y_pred,
            mode='lines',
            name='Predicted',
            line=dict(color='red', width=2, dash='dash')
        )
    )
    
    # Add future predictions if provided
    if future_predictions is not None:
        future_x = list(range(len(y_true), len(y_true) + len(future_predictions)))
        fig.add_trace(
            go.Scatter(
                x=future_x,
                y=future_predictions,
                mode='lines+markers',
                name='Future Predictions',
                line=dict(color='green', width=2),
                marker=dict(size=6)
            )
        )
    
    fig.update_layout(
        title=f'{model_name} Model Performance',
        xaxis_title='Time Steps',
        yaxis_title='Value',
        hovermode='x unified',
        height=400
    )
    
    return fig

def create_training_history_plot(history, model_name):
    """Create interactive plot for training history (loss curves)"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Training & Validation Loss', 'Training & Validation MAE']
    )
    
    epochs = range(1, len(history.history['loss']) + 1)
    
    # Loss plot
    fig.add_trace(
        go.Scatter(
            x=list(epochs),
            y=history.history['loss'],
            mode='lines',
            name='Training Loss',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    if 'val_loss' in history.history:
        fig.add_trace(
            go.Scatter(
                x=list(epochs),
                y=history.history['val_loss'],
                mode='lines',
                name='Validation Loss',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=1
        )
    
    # MAE plot
    if 'mae' in history.history:
        fig.add_trace(
            go.Scatter(
                x=list(epochs),
                y=history.history['mae'],
                mode='lines',
                name='Training MAE',
                line=dict(color='green', width=2),
                showlegend=True
            ),
            row=1, col=2
        )
    
    if 'val_mae' in history.history:
        fig.add_trace(
            go.Scatter(
                x=list(epochs),
                y=history.history['val_mae'],
                mode='lines',
                name='Validation MAE',
                line=dict(color='orange', width=2, dash='dash'),
                showlegend=True
            ),
            row=1, col=2
        )
    
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss (MSE)", row=1, col=1)
    fig.update_yaxes(title_text="MAE", row=1, col=2)
    
    fig.update_layout(
        title=f'{model_name} - Training History',
        height=350,
        hovermode='x unified'
    )
    
    return fig

def perform_seasonal_decomposition(data, column_name, period=30):
    """Perform seasonal decomposition and create visualization"""
    try:
        # Ensure sufficient data for decomposition
        if len(data) < 2 * period:
            return None, f"Need at least {2 * period} data points for decomposition with period={period}"
        
        # Perform decomposition
        decomposition = seasonal_decompose(
            data[column_name].values,
            model='additive',
            period=period,
            extrapolate_trend='freq'
        )
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual'],
            vertical_spacing=0.08
        )
        
        # Original data
        fig.add_trace(
            go.Scatter(
                y=data[column_name].values,
                mode='lines',
                name='Original',
                line=dict(color='blue', width=1.5)
            ),
            row=1, col=1
        )
        
        # Trend
        fig.add_trace(
            go.Scatter(
                y=decomposition.trend,
                mode='lines',
                name='Trend',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        # Seasonal
        fig.add_trace(
            go.Scatter(
                y=decomposition.seasonal,
                mode='lines',
                name='Seasonal',
                line=dict(color='orange', width=1.5)
            ),
            row=3, col=1
        )
        
        # Residual
        fig.add_trace(
            go.Scatter(
                y=decomposition.resid,
                mode='lines',
                name='Residual',
                line=dict(color='red', width=1)
            ),
            row=4, col=1
        )
        
        fig.update_layout(
            title=f'Seasonal Decomposition - {column_name}',
            height=800,
            showlegend=False,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Time", row=4, col=1)
        
        return fig, None
        
    except Exception as e:
        return None, f"Error in decomposition: {str(e)}"

def main():
    # Initialize session state for data persistence
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'data_source' not in st.session_state:
        st.session_state.data_source = None
    if 'saved_models' not in st.session_state:
        st.session_state.saved_models = {}
    if 'saved_scalers' not in st.session_state:
        st.session_state.saved_scalers = {}
    
    # Main header
    st.markdown('<div class="main-header">🟢 Multi-Model Time Series Forecasting</div>', unsafe_allow_html=True)
    
    # Sidebar for parameters
    st.sidebar.header("📊 Model Configuration")
    
    # File upload section
    st.header("📁 Data Upload")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload a CSV file with time series data",
            type=['csv'],
            help="Upload a CSV file with at least one numeric column for forecasting"
        )
    
    with col2:
        use_demo = st.button("Use Demo Data", type="secondary")
    
    # Load data and persist in session state
    if uploaded_file is not None:
        try:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.session_state.data_source = "uploaded"
            st.success(f"✅ File uploaded successfully! Shape: {st.session_state.data.shape}")
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return
    elif use_demo:
        st.session_state.data = create_demo_data()
        st.session_state.data_source = "demo"
        st.success("✅ Demo data loaded successfully!")
        st.info("📊 Demo data contains 500 days of synthetic time series with trend and seasonality")
    
    # Get data from session state
    data = st.session_state.data
    
    if data is not None:
        # Display data preview
        st.subheader("📋 Data Preview")
        st.dataframe(data.head(10))
        
        # Select target column
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            st.error("❌ No numeric columns found in the data!")
            return
        
        target_column = st.selectbox(
            "Select target column for forecasting",
            numeric_columns,
            index=0
        )
        
        # Display basic statistics
        st.subheader("📈 Data Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(data))
        with col2:
            st.metric("Mean Value", f"{data[target_column].mean():.2f}")
        with col3:
            st.metric("Std Deviation", f"{data[target_column].std():.2f}")
        with col4:
            st.metric("Missing Values", data[target_column].isnull().sum())
        
        # Plot original data
        fig_original = px.line(
            data, 
            y=target_column,
            title=f"Original Time Series - {target_column}",
            labels={'index': 'Time Steps', target_column: 'Value'}
        )
        st.plotly_chart(fig_original, use_container_width=True)
        
        # Seasonal decomposition analysis
        st.subheader("🔍 Seasonal Decomposition Analysis")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            show_decomposition = st.checkbox("Show Decomposition", value=False)
        
        if show_decomposition:
            # Check if dataset is large enough for decomposition
            min_period = 7
            max_period_calc = len(data) // 3
            
            if max_period_calc < min_period:
                st.warning(f"⚠️ Dataset too small for seasonal decomposition. Need at least {min_period * 3} data points.")
            else:
                with col2:
                    decomp_period = st.slider(
                        "Seasonal Period",
                        min_value=min_period,
                        max_value=min(365, max_period_calc),
                        value=min(30, max_period_calc),
                        help="The period of the seasonal component (e.g., 7 for weekly, 30 for monthly)"
                    )
                
                # Ensure period is valid for current data length
                if decomp_period >= len(data) // 2:
                    st.warning(f"⚠️ Period ({decomp_period}) is too large for dataset size ({len(data)}). Maximum period is {len(data) // 2 - 1}.")
                else:
                    decomp_fig, error = perform_seasonal_decomposition(data, target_column, decomp_period)
                    
                    if decomp_fig is not None:
                        st.plotly_chart(decomp_fig, use_container_width=True)
                        st.info("📊 Seasonal decomposition breaks down the time series into trend, seasonal, and residual components.")
                    else:
                        st.warning(f"⚠️ {error}")
        
        # Model configuration in sidebar
        st.sidebar.subheader("🔧 Training Parameters")
        
        sequence_length = st.sidebar.slider(
            "Sequence Length",
            min_value=10,
            max_value=100,
            value=50,
            help="Number of previous time steps to use for prediction"
        )
        
        epochs = st.sidebar.number_input(
            "Epochs",
            min_value=10,
            max_value=500,
            value=50,
            help="Number of training epochs"
        )
        
        batch_size = st.sidebar.number_input(
            "Batch Size",
            min_value=16,
            max_value=128,
            value=32,
            help="Training batch size"
        )
        
        # Model selection
        st.sidebar.subheader("🤖 Model Selection")
        selected_models = st.sidebar.multiselect(
            "Select Models to Train",
            ["ANN", "RNN", "LSTM", "GRU"],
            default=["LSTM"],
            help="Choose which models to train and compare"
        )
        
        if not selected_models:
            st.sidebar.error("Please select at least one model!")
            return
        
        # Multi-step forecasting options
        st.sidebar.subheader("🔮 Forecasting Options")
        enable_multistep = st.sidebar.checkbox(
            "Enable Multi-Step Forecast",
            value=False,
            help="Predict multiple future values"
        )
        
        future_steps = 5
        if enable_multistep:
            future_steps = st.sidebar.number_input(
                "Number of Future Steps",
                min_value=1,
                max_value=50,
                value=5,
                help="Number of future values to predict"
            )
        
        # Training button
        if st.sidebar.button("🚀 Train and Forecast", type="primary"):
            if len(data) < sequence_length + 50:  # Minimum data requirement
                st.error(f"❌ Insufficient data! Need at least {sequence_length + 50} records.")
                return
            
            # Preprocess data
            with st.spinner("🔄 Preprocessing data..."):
                X, y, scaler, original_values = preprocess_data(data, sequence_length, target_column)
                
                # Split data
                train_size = int(len(X) * 0.8)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]
            
            # Results storage
            results = {}
            model_performances = []
            
            # Train selected models
            for model_name in selected_models:
                st.subheader(f"🤖 Training {model_name} Model")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Build model
                    status_text.text(f"Building {model_name} model...")
                    progress_bar.progress(25)
                    
                    if model_name == "ANN":
                        model = build_ann_model(sequence_length)
                        train_X = X_train
                        test_X = X_test
                    elif model_name == "RNN":
                        model = build_rnn_model(sequence_length)
                        train_X = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                        test_X = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                    elif model_name == "LSTM":
                        model = build_lstm_model(sequence_length)
                        train_X = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                        test_X = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                    elif model_name == "GRU":
                        model = build_gru_model(sequence_length)
                        train_X = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                        test_X = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                    
                    # Train model
                    status_text.text(f"Training {model_name} model...")
                    progress_bar.progress(50)
                    
                    trained_model, history = train_model(
                        model, train_X, y_train, epochs, batch_size
                    )
                    
                    # Make predictions
                    status_text.text(f"Making predictions with {model_name}...")
                    progress_bar.progress(75)
                    
                    y_pred = trained_model.predict(test_X, verbose=0)
                    
                    # Calculate RMSE
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    
                    # Inverse transform for plotting
                    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
                    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                    
                    # Future predictions
                    future_preds_inv = None
                    if enable_multistep:
                        last_sequence = X[-1]  # Last sequence from the data
                        future_preds_inv = predict_future_steps(
                            trained_model, last_sequence, scaler, future_steps, model_name
                        )
                    
                    # Store results
                    results[model_name] = {
                        'model': trained_model,
                        'y_test': y_test_inv,
                        'y_pred': y_pred_inv,
                        'rmse': rmse,
                        'future_predictions': future_preds_inv,
                        'history': history
                    }
                    
                    # Auto-save model to session state
                    st.session_state.saved_models[model_name] = trained_model
                    st.session_state.saved_scalers[model_name] = scaler
                    
                    model_performances.append({
                        'Model': model_name,
                        'RMSE': rmse,
                        'MAE': np.mean(np.abs(y_test_inv - y_pred_inv))
                    })
                    
                    progress_bar.progress(100)
                    status_text.text(f"✅ {model_name} training completed!")
                    
                    # Show training metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(f"{model_name} RMSE", f"{rmse:.4f}")
                    with col2:
                        st.metric(f"{model_name} MAE", f"{np.mean(np.abs(y_test_inv - y_pred_inv)):.4f}")
                    
                except Exception as e:
                    st.error(f"❌ Error training {model_name}: {str(e)}")
                    continue
            
            if results:
                # Display results
                st.header("📊 Model Comparison Results")
                
                # Performance comparison table
                st.subheader("🏆 Model Performance Comparison")
                performance_df = pd.DataFrame(model_performances)
                performance_df = performance_df.sort_values('RMSE')
                
                # Highlight best model
                st.dataframe(
                    performance_df.style.highlight_min(subset=['RMSE'], color='lightgreen'),
                    use_container_width=True
                )
                
                # Plots for each model
                st.subheader("📈 Prediction Visualizations")
                
                for model_name, result in results.items():
                    st.write(f"### {model_name} Model Results")
                    
                    # Create two columns for plots
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        # Create prediction plot
                        fig = create_prediction_plot(
                            result['y_test'],
                            result['y_pred'],
                            model_name,
                            result['future_predictions'] if enable_multistep else None
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Create training history plot
                        if result['history'] is not None:
                            history_fig = create_training_history_plot(result['history'], model_name)
                            st.plotly_chart(history_fig, use_container_width=True)
                    
                    # Show future predictions if enabled
                    if enable_multistep and result['future_predictions'] is not None:
                        st.write(f"**🔮 Next {future_steps} Predicted Values ({model_name}):**")
                        future_df = pd.DataFrame({
                            'Step': range(1, future_steps + 1),
                            'Predicted Value': result['future_predictions']
                        })
                        st.dataframe(future_df, use_container_width=True)
                
                # Download predictions
                st.header("💾 Download Predictions")
                
                for model_name, result in results.items():
                    # Prepare download data
                    download_data = {
                        'Actual': result['y_test'],
                        'Predicted': result['y_pred']
                    }
                    
                    if enable_multistep and result['future_predictions'] is not None:
                        # Add future predictions with NaN for actual values
                        future_actual = [np.nan] * len(result['future_predictions'])
                        download_data['Actual'] = np.concatenate([download_data['Actual'], future_actual])
                        download_data['Predicted'] = np.concatenate([download_data['Predicted'], result['future_predictions']])
                    
                    download_df = pd.DataFrame(download_data)
                    
                    # Convert to CSV
                    csv_buffer = io.StringIO()
                    download_df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    st.download_button(
                        label=f"📥 Download {model_name} Predictions",
                        data=csv_data,
                        file_name=f"{model_name}_predictions.csv",
                        mime="text/csv",
                        key=f"download_{model_name}"
                    )
                
                st.success("🎉 Training and forecasting completed successfully!")
        
        # Model Management Section
        if st.session_state.saved_models:
            st.header("💾 Saved Models")
            
            st.info(f"You have {len(st.session_state.saved_models)} saved model(s) in this session that can be reused without retraining.")
            
            # Display saved models
            saved_models_df = pd.DataFrame({
                'Model Name': list(st.session_state.saved_models.keys()),
                'Status': ['Ready' for _ in st.session_state.saved_models]
            })
            st.dataframe(saved_models_df, use_container_width=True)
            
            # Use saved model for inference
            st.subheader("🔮 Use Saved Model for Forecasting")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                selected_saved_model = st.selectbox(
                    "Select a saved model",
                    list(st.session_state.saved_models.keys())
                )
            
            with col2:
                forecast_steps_saved = st.number_input(
                    "Future Steps",
                    min_value=1,
                    max_value=50,
                    value=5,
                    key="forecast_steps_saved"
                )
            
            with col3:
                if st.button("📊 Generate Forecast", type="primary"):
                    if selected_saved_model and data is not None:
                        try:
                            # Get saved model and scaler
                            saved_model = st.session_state.saved_models[selected_saved_model]
                            saved_scaler = st.session_state.saved_scalers[selected_saved_model]
                            
                            # Prepare data
                            values = data[target_column].values.reshape(-1, 1)
                            scaled_values = saved_scaler.transform(values)
                            
                            # Get last sequence
                            sequence_len = saved_model.input_shape[1] if len(saved_model.input_shape) > 2 else saved_model.input_shape[1]
                            last_sequence = scaled_values[-sequence_len:, 0]
                            
                            # Generate forecast
                            future_preds = predict_future_steps(
                                saved_model,
                                last_sequence,
                                saved_scaler,
                                forecast_steps_saved,
                                selected_saved_model
                            )
                            
                            # Display results
                            st.success(f"✅ Generated {forecast_steps_saved} future predictions using {selected_saved_model}")
                            
                            # Create visualization
                            fig = go.Figure()
                            
                            # Historical data (last 100 points)
                            hist_data = data[target_column].values[-100:]
                            fig.add_trace(
                                go.Scatter(
                                    y=hist_data,
                                    mode='lines',
                                    name='Historical',
                                    line=dict(color='blue', width=2)
                                )
                            )
                            
                            # Future predictions
                            future_x = list(range(len(hist_data), len(hist_data) + len(future_preds)))
                            fig.add_trace(
                                go.Scatter(
                                    x=future_x,
                                    y=future_preds,
                                    mode='lines+markers',
                                    name='Forecast',
                                    line=dict(color='green', width=2),
                                    marker=dict(size=8)
                                )
                            )
                            
                            fig.update_layout(
                                title=f'Forecast using {selected_saved_model}',
                                xaxis_title='Time Steps',
                                yaxis_title='Value',
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show forecast table
                            forecast_df = pd.DataFrame({
                                'Step': range(1, forecast_steps_saved + 1),
                                'Predicted Value': future_preds
                            })
                            st.dataframe(forecast_df, use_container_width=True)
                            
                            # Download button
                            csv_buffer = io.StringIO()
                            forecast_df.to_csv(csv_buffer, index=False)
                            csv_data = csv_buffer.getvalue()
                            
                            st.download_button(
                                label=f"📥 Download Forecast",
                                data=csv_data,
                                file_name=f"{selected_saved_model}_forecast.csv",
                                mime="text/csv",
                                key=f"download_forecast_{selected_saved_model}"
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Error generating forecast: {str(e)}")
                    else:
                        st.warning("⚠️ Please load data first!")
            
            st.markdown("---")
            
            # Option to clear saved models
            if st.button("🗑️ Clear All Saved Models"):
                st.session_state.saved_models = {}
                st.session_state.saved_scalers = {}
                st.rerun()
    
    else:
        st.info("👆 Please upload a CSV file or use demo data to get started.")
        
        # Show instructions
        st.markdown("""
        ### 📋 How to use this app:
        
        1. **Upload Data**: Upload a CSV file with time series data or use the demo data
        2. **Select Target**: Choose the numeric column you want to forecast
        3. **Configure Models**: Adjust parameters in the sidebar (sequence length, epochs, etc.)
        4. **Select Models**: Choose which models to train (ANN, RNN, LSTM)
        5. **Enable Forecasting**: Optionally enable multi-step future predictions
        6. **Train & Forecast**: Click the button to start training
        7. **View Results**: Analyze predictions, compare models, and download results
        
        ### 📊 Model Information:
        - **ANN (Artificial Neural Network)**: Feedforward network for short-term predictions
        - **RNN (Recurrent Neural Network)**: Vanilla RNN for capturing sequential dependencies  
        - **LSTM (Long Short-Term Memory)**: Advanced RNN for long-term dependencies and trends
        - **GRU (Gated Recurrent Unit)**: Efficient RNN variant with faster training than LSTM
        """)

if __name__ == "__main__":
    main()

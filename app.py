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
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
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
            # For RNN/LSTM, reshape to (1, time_steps, 1)
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

def main():
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
    
    # Load data
    data = None
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Shape: {data.shape}")
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return
    elif use_demo:
        data = create_demo_data()
        st.success("✅ Demo data loaded successfully!")
        st.info("📊 Demo data contains 500 days of synthetic time series with trend and seasonality")
    
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
            ["ANN", "RNN", "LSTM"],
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
                    else:  # LSTM
                        model = build_lstm_model(sequence_length)
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
                    
                    # Create prediction plot
                    fig = create_prediction_plot(
                        result['y_test'],
                        result['y_pred'],
                        model_name,
                        result['future_predictions'] if enable_multistep else None
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
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
        """)

if __name__ == "__main__":
    main()

# Fraud Detection Production

An advanced fraud detection system using TensorFlow's autoencoder architecture with optimized CPU-GPU parallel processing pipelines. This production-ready system extends previous work by implementing efficient ETL pipelines and TensorFlow Serving for deployment.

## Project Architecture

```
├── checkpoints/     # Model training checkpoints
├── data/           
│   ├── processed/   # Preprocessed datasets
│   └── raw/        # Original data files
├── deployment/      # TensorFlow Serving API configurations
├── model-export/    # Exported SavedModel files
├── notebooks/       # Data exploration and prototypes
├── src/            # Source code
└── summary/        # Training and validation summaries
```

## Technical Requirements

- TensorFlow v1.13.1
- TensorFlow Serving API v1.13.1
- Python 3.6+
- Docker (for containerized deployment)

## Deployment Options

### Option 1: Standard Deployment

1. Install TensorFlow Serving and configure port:
```bash
tensorflow_model_server --port=8500 \
                       --model_name=anamoly_detection \
                       --model_base_path=$HOME/Desktop/Fraud-Detection-Production-master/model-export/anamoly_detection/
```

2. Run the client:
```bash
python client.py
```

### Option 2: Docker Deployment (Recommended)

1. Pull TensorFlow Serving image:
```bash
docker pull tensorflow/serving
```

2. Create container:
```bash
docker create -p 8500:8500 \
    -e MODEL_NAME=anamoly_detection \
    --mount type=bind,source=$HOME/Desktop/Fraud-Detection-Production/model-export/anamoly_detection,target=/models/anamoly_detection \
    --name=my_container1 \
    tensorflow/serving
```

3. Start container:
```bash
docker start my_container1
```

4. Run client:
```bash
python client.py
```

## Key Features

- Optimized ETL pipelines for efficient data processing
- CPU-GPU parallel processing for enhanced performance
- Production-ready autoencoder model for fraud detection
- TensorFlow Serving integration for scalable deployment
- Docker containerization support
- AWS deployment configuration (optional)

## Performance Notes

- Leverages TensorFlow's parallel processing capabilities
- Optimized ETL reduces processing overhead
- Docker containerization ensures consistent deployment
- AWS deployment available for production scaling

## Cloud Deployment

While AWS deployment is supported, note that costs can be significant. Docker deployment is recommended for development and testing environments.

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

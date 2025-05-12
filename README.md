
# RealTime-ObjectDetection-AV

A real-time object detection system designed for autonomous vehicles, leveraging deep learning to identify pedestrians, vehicles, traffic signs, and obstacles under diverse conditions. The project encompasses model training, MLOps integration for monitoring, and real-world testing to ensure safety and efficiency in self-driving applications.

##  Features

- **Real-Time Detection**: Processes live video feeds to detect multiple object classes pertinent to autonomous driving.
- **Deep Learning Integration**: Utilizes state-of-the-art models for accurate object recognition.
- **MLOps Monitoring**: Incorporates tools for model performance tracking and system diagnostics.
- **Comprehensive Testing**: Includes real-world scenario evaluations to validate system reliability.

##  Project Structure

```
RealTime-ObjectDetection-AV/
├── Deployment/                    # Deployment files 
├── Exported Model/                # Pre-trained and custom-trained models
├── Presentation/                  # Presentation           
├── AV_Project.ipynb               # Python notebook
└── README.md                      # Project documentation
```

##  Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/mohamed-khaledd/RealTime-ObjectDetection-AV.git
   cd RealTime-ObjectDetection-AV
   ```

2. **Create a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

##  Usage

1. **Prepare the dataset**:

   Place your training and validation datasets in the `data/` directory, following the required structure.

2. **Train the model**:

   ```bash
   python scripts/train.py --config configs/train_config.yaml
   ```

3. **Monitor performance**:

   Utilize integrated MLOps tools to track model metrics and system logs.

##  Testing

Execute the test suite to ensure all components function as expected:

```bash
python -m unittest discover tests
```

##  License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

##  Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

##  Contact

For questions or suggestions, feel free to open an issue or contact [Mohamed Khaled](https://github.com/mohamed-khaledd).

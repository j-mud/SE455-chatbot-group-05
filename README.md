# Agriculture SLM — Edge Intelligence Project

## UG-5: On-Device Small Language Models for Edge Intelligence

This project is a smart agriculture edge intelligence system that uses environmental sensor readings such as temperature, humidity, and gas level to detect agricultural conditions. The system compares lightweight models and supports local decision-making, command processing, and edge-based inference.

The purpose of the project is to show how small and efficient models can be used on edge devices instead of depending fully on cloud services. This helps reduce latency, improve privacy, and support offline or low-connectivity environments.

---

## Project Structure

```text
agriculture_slm/
├── main.py
├── requirements.txt
├── data/
│   └── generate_dataset.py
├── models/
│   ├── lightweight_models.py
│   └── slm_distilbert.py
├── experiments/
│   ├── run_experiments.py
│   ├── cloud_baseline.py
│   └── pi_deploy.py
└── utils/
    └── intent_processor.py
```

---

## Main Files

| File                             | Description                                                                               |
| -------------------------------- | ----------------------------------------------------------------------------------------- |
| `main.py`                        | Runs the main pipeline for dataset generation, model training, and experiments            |
| `requirements.txt`               | Lists the required Python dependencies                                                    |
| `data/generate_dataset.py`       | Creates the agriculture dataset used for training and testing                             |
| `models/lightweight_models.py`   | Implements lightweight models such as Decision Tree, Random Forest, and Gradient Boosting |
| `models/slm_distilbert.py`       | Optional DistilBERT-based small language model implementation                             |
| `experiments/run_experiments.py` | Runs the experiments and generates model comparison results                               |
| `experiments/cloud_baseline.py`  | Compares local edge inference with cloud-based processing                                 |
| `experiments/pi_deploy.py`       | Runs inference on the Raspberry Pi using simulated or real sensor data                    |
| `utils/intent_processor.py`      | Handles command processing and local decision logic                                       |

---

## Setup

The project can be opened using any Python-supported development environment, such as VS Code, PyCharm, or a terminal.

### Create a Virtual Environment

```bash
python -m venv venv
```

### Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Mac/Linux

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

DistilBERT is included as an optional small language model experiment, while the lightweight models can run separately. `torch` and `transformers` are only needed if you want to run the DistilBERT file. The other models can run without them.

---

## Running the Main Pipeline

The full pipeline can be started with:

```bash
python main.py
```

This step runs dataset generation, lightweight model training, and experiment evaluation.

Expected generated outputs include:

```text
agriculture_dataset.csv
saved_models/
results/full_comparison_table.csv
```

---

## Optional DistilBERT Model

The DistilBERT-based SLM can be trained separately using:

```bash
python models/slm_distilbert.py
```

This step is optional and may take longer depending on the device. The printed accuracy and latency values can be added to the comparison results.

---

## Cloud Baseline Comparison

The cloud baseline is used to compare local edge inference with cloud-based processing.

To start the server:

```bash
python experiments/cloud_baseline.py server
```

To run the client:

```bash
python experiments/cloud_baseline.py client --host <your_laptop_ip>
```

This comparison helps evaluate the difference between local processing and cloud-based inference in terms of response time and deployment trade-offs.

---

## Raspberry Pi Deployment

For Raspberry Pi deployment, the project folder should be available on the Raspberry Pi environment. The required lightweight dependencies are:

```bash
pip install numpy pandas scikit-learn joblib psutil requests
```

### Simulated Sensor Data

```bash
python experiments/pi_deploy.py --mode simulate --n 30
```

### Real ESP32 Serial Data

```bash
python experiments/pi_deploy.py --mode serial --port /dev/ttyUSB0
```

### Cloud vs Edge Latency Test

```bash
python experiments/cloud_baseline.py client --host <your_laptop_ip>
```

The Raspberry Pi inference results are saved in:

```text
results/pi_inference_log.csv
```

---

## ESP32 Serial Input Format

The ESP32 sends sensor readings in comma-separated format:

```text
temperature,humidity,gas_level
```

Example readings:

```text
25.3,67.2,412
41.1,28.5,395
```

Each line represents one sensor reading.

---

## Expected Results

| Model             | Accuracy | Average Latency | Memory Usage |
| ----------------- | -------: | --------------: | -----------: |
| Decision Tree     |     ~78% |           <5 ms |       <10 MB |
| Random Forest     |     ~88% |           ~8 ms |       ~30 MB |
| Gradient Boosting |     ~90% |          ~15 ms |       ~50 MB |
| DistilBERT        |  ~85–92% |      ~80–150 ms |      ~250 MB |
| Cloud LLM API     |     ~94% |     ~300–600 ms |          N/A |

The expected results show that lightweight local models can provide strong performance with much lower latency than cloud-based inference. Random Forest provides a good balance between accuracy, speed, and memory usage, making it suitable for edge deployment. Cloud-based models may achieve slightly higher accuracy, but they usually require higher latency and external data transmission.

---

## Evaluation Metrics

The main evaluation metrics are:

* Accuracy
* Average inference latency
* Memory usage
* Edge deployment feasibility
* Cloud vs edge response time
* Privacy and offline capability

---

## Limitations

* Performance depends on the selected hardware platform.
* ESP32 has stricter memory and processing limits than Raspberry Pi.
* The natural language interaction is limited to simple commands.
* Real-world agricultural testing may require longer sensor data collection.
* Energy consumption needs more detailed measurement in future testing.

---

## Future Work

The system can be improved later by:

* Adding Arabic language support
* Integrating more agricultural sensors such as soil moisture and light intensity
* Applying quantization to reduce model size
* Improving command processing and natural language interaction
* Testing the system over longer real-world deployment periods
* Exploring a hybrid edge-cloud architecture for complex tasks.

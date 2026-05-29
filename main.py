import os
import sys

print("="*60)
print("Agriculture SLM — Edge Intelligence Project")
print("="*60)


print("\n[1/3] Generating dataset...")
from data.generate_dataset import generate_dataset
df = generate_dataset(output_csv="agriculture_dataset.csv")


print("\n[2/3] Training lightweight models...")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

LABEL_CLASSES = ["NORMAL", "HIGH_TEMP", "HIGH_HUMIDITY", "GAS_ANOMALY", "COMBINED_STRESS"]
FEATURES = ["temperature", "humidity", "gas_level"]

df_data = pd.read_csv("agriculture_dataset.csv")
le = LabelEncoder()
le.fit(LABEL_CLASSES)
X = df_data[FEATURES].values
y = le.transform(df_data["label"].values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from models.lightweight_models import train_all_models, evaluate_model
import pandas as pd

os.makedirs("results", exist_ok=True)
models = train_all_models(X_train, y_train, save_dir="saved_models")

results = []
for name, model in models.items():
    r = evaluate_model(model, X_test, y_test, le, name)
    results.append(r)

pd.DataFrame(results).to_csv("results/lightweight_models_results.csv", index=False)


print("\n[3/3] Running experiments and generating comparison table...")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from experiments.run_experiments import run_experiments, generate_paper_table
df_results = run_experiments()
generate_paper_table()

print("\n" + "="*60)
print("ALL DONE.")
print("="*60)
print("\nOutput files:")
print("  agriculture_dataset.csv              — your labeled dataset")
print("  saved_models/DecisionTree.pkl        — trained model")
print("  saved_models/RandomForest.pkl        — trained model (use this on Pi)")
print("  saved_models/GradientBoosting.pkl    — trained model")
print("  results/lightweight_models_results.csv")
print("  results/full_comparison_table.csv    — your Table 1 for the paper")
print("\nNext steps:")
print("  1. Run models/slm_distilbert.py to add DistilBERT results")
print("  2. Copy project to Raspberry Pi, run: python experiments/pi_deploy.py")
print("  3. Run cloud baseline: python experiments/cloud_baseline.py server (laptop)")
print("     Then on Pi: python experiments/cloud_baseline.py client --host <laptop_ip>")
print("  4. Fill DistilBERT + cloud latency into results/full_comparison_table.csv")

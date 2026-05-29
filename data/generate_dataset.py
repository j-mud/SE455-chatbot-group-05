import pandas as pd
import numpy as np

def generate_dataset(
    crop_csv="data/Crop_recommendation.csv",
    iot_csv="data/iot_telemetry_data.csv",
    output_csv="agriculture_dataset.csv"
):

    crop_df = pd.read_csv(crop_csv)
    print(f"Crop dataset loaded: {len(crop_df)} rows")
    print(f"  Columns: {list(crop_df.columns)}")

    def label_from_crop_env(row):
        t = row["temperature"]
        h = row["humidity"]
        if t > 38:
            return "HIGH_TEMP"
        elif h > 85:
            return "HIGH_HUMIDITY"
        elif t >= 18 and t <= 32 and h >= 50 and h <= 80:
            return "NORMAL"
        else:
            return "NORMAL"  

    crop_df["stress_label"] = crop_df.apply(label_from_crop_env, axis=1)
    crop_df["gas_level"] = 400  

    crop_prepared = pd.DataFrame({
        "temperature": crop_df["temperature"],
        "humidity":    crop_df["humidity"],
        "gas_level":   crop_df["gas_level"],
        "label":       crop_df["stress_label"]
    })


    iot_df = pd.read_csv(iot_csv)
    print(f"\nIoT telemetry dataset loaded: {len(iot_df)} rows")
    print(f"  Columns: {list(iot_df.columns)}")


    iot_df = iot_df.rename(columns={"temp": "temperature"})

    lpg_min = iot_df["lpg"].min()
    lpg_max = iot_df["lpg"].max()
    iot_df["gas_level"] = (
        (iot_df["lpg"] - lpg_min) / (lpg_max - lpg_min) * 900 + 300
    ).round(0).astype(int)

def label_from_iot(row):
    t   = row["temperature"]
    h   = row["humidity"]
    gas = row["gas_level"]
    
    stress_count = (t > 35) + (h > 80) + (gas > 700)
    if stress_count >= 2:
        return "COMBINED_STRESS"
    elif gas > 700:
        return "GAS_ANOMALY"
    elif t > 35:
        return "HIGH_TEMP"
    elif h > 80:
        return "HIGH_HUMIDITY"
    else:
        return "NORMAL"

    iot_df["label"] = iot_df.apply(label_from_iot, axis=1)

    iot_prepared = pd.DataFrame({
        "temperature": iot_df["temperature"],
        "humidity":    iot_df["humidity"],
        "gas_level":   iot_df["gas_level"],
        "label":       iot_df["label"]
    }).dropna()



    combined = pd.concat([crop_prepared, iot_prepared], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)


    combined["text_input"] = combined.apply(
        lambda r: f"temperature {r['temperature']:.1f} humidity {r['humidity']:.1f} gas {r['gas_level']:.0f}",
        axis=1
    )

    rng = np.random.default_rng(42)
    combined["temperature"] = combined["temperature"] + rng.normal(0, 0.5, len(combined))
    combined["humidity"]    = combined["humidity"]    + rng.normal(0, 1.0, len(combined))
    combined["gas_level"]   = combined["gas_level"]   + rng.normal(0, 10,  len(combined))

    combined.to_csv(output_csv, index=False)
    print(f"\nCombined dataset saved: {output_csv}")
    print(f"Total rows: {len(combined)}")
    print(f"Class distribution:\n{combined['label'].value_counts()}")
    return combined


if __name__ == "__main__":
    df = generate_dataset()
    print("\nSample:")
    print(df.head())

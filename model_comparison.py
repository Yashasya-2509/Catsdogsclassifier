import pandas as pd

def get_model_results():

    data = {
        "Model": [
            "Custom CNN",
            "MobileNetV2"
        ],
        "Accuracy": [
            92.4,
            98.7
        ],
        "Parameters": [
            2500000,
            2257985
        ]
    }

    return pd.DataFrame(data)

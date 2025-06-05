import os
import pandas as pd

def save_metrics_and_accuracies(class_accuracies, OA, AA, Kappa, Train_time, Test_time, dataset_name, output_folder="output"):

    os.makedirs(output_folder, exist_ok=True)

    df_class_acc = pd.DataFrame([class_accuracies], columns=[f"Class_{i + 1}_Accuracy" for i in range(len(class_accuracies))])
    df_metrics = pd.DataFrame({
        'Overall Accuracy (OA)': [OA],
        'Average Accuracy (AA)': [AA],
        'Kappa Coefficient': [Kappa]
    })
    df_time = pd.DataFrame({
        'Training Time': [Train_time],
        'Testing Time': [Test_time]
    })

    df_final = pd.concat([df_class_acc, df_metrics, df_time], axis=1)

    output_file = os.path.join(output_folder, f"{dataset_name}_acc.csv")

    df_final.T.to_csv(output_file, header=False)

    print(f"Results saved to {output_file}")

    return output_file

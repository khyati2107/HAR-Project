import argparse
from preprocessing import UCIHARPreprocessor, plot_dataset_overview
from clustering import run_clustering
from classification import train_and_evaluate_models
from neural_network import run_neural_network


def main():
    parser = argparse.ArgumentParser(description="UCI HAR Dataset Pipeline")
    parser.add_argument(
        '--data_path',
        type=str,
        default='UCI HAR Dataset',
        help='Path to the UCI HAR Dataset ZIP file'
    )
    args = parser.parse_args()

    # 1. Preprocessing
    preprocessor = UCIHARPreprocessor(sampling_rate=50, window_size=2.56, overlap=0.5)
    X_train, X_test, y_train, y_test = preprocessor.full_preprocessing_pipeline(args.data_path)

    if X_train is None:
        print("Preprocessing failed. Exiting.")
        return

    processed_data = {
        'X_train':        X_train,
        'X_test':         X_test,
        'y_train':        y_train,
        'y_test':         y_test,
        'activity_labels': preprocessor.activity_labels,
        'scaler':         preprocessor.scaler,
        'pca':            preprocessor.pca,
        'common_features': preprocessor.common_features
    }

    # 2. Dataset overview plots
    print("\nGenerating dataset overview...")
    plot_dataset_overview(X_train, X_test, y_train, y_test, preprocessor.activity_labels)

    # 3. Clustering
    print("\nRunning clustering...")
    run_clustering(X_train, n_clusters=6)

    # 4. Classification
    print("\nRunning classification models...")
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    print("\nFinal Accuracy Summary:")
    for model_name, acc in results.items():
        print(f"  {model_name}: {acc*100:.2f}%")

    # 5. Neural Network
    print("\nRunning PyTorch Neural Network...")
    run_neural_network(processed_data)


if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import zipfile

os.makedirs('outputs', exist_ok=True)


class UCIHARPreprocessor:
    def __init__(self, sampling_rate=50, window_size=2.56, overlap=0.5):
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.overlap = overlap
        self.scaler = StandardScaler()
        self.pca = None
        self.activity_labels = {
            1: 'WALKING',
            2: 'WALKING_UPSTAIRS',
            3: 'WALKING_DOWNSTAIRS',
            4: 'SITTING',
            5: 'STANDING',
            6: 'LAYING'
        }
        self.common_features = None

    def find_har_files(self, base_path):
        possible_paths = [
            base_path,
            os.path.join(base_path, 'UCI HAR Dataset'),
            os.path.join(base_path, 'uci_har_dataset'),
            os.path.join(base_path, 'UCI-HAR-Dataset'),
        ]
        for path in possible_paths:
            train_file = os.path.join(path, 'train', 'X_train.txt')
            if os.path.exists(train_file):
                print(f"Found dataset at: {path}")
                return path

        print("Searching for dataset files recursively...")
        for root, dirs, files in os.walk(base_path):
            if 'X_train.txt' in files and 'train' in root:
                print(f"Found dataset at: {root}")
                return os.path.dirname(root) if 'train' in root else root
        return None

    def load_uci_har_dataset(self, data_path):
        if data_path.endswith('.zip'):
            if not os.path.exists(data_path):
                print(f"ZIP file not found at: {data_path}")
                return None, None, None, None, None
            print(f"Extracting {data_path}...")
            with zipfile.ZipFile(data_path, 'r') as zip_ref:
                zip_ref.extractall('extracted_dataset')
            base_path = self.find_har_files('extracted_dataset')
        else:
            base_path = self.find_har_files(data_path)

        if base_path is None:
            print("Could not find the dataset structure.")
            for root, dirs, files_list in os.walk(data_path):
                for file in files_list:
                    if file.endswith('.txt'):
                        print(f"Found: {os.path.join(root, file)}")
            return None, None, None, None, None

        try:
            train_path = os.path.join(base_path, 'train', 'X_train.txt')
            test_path = os.path.join(base_path, 'test', 'X_test.txt')

            print(f"Loading training data from: {train_path}")
            print(f"Loading test data from: {test_path}")

            X_train = pd.read_csv(train_path, sep=r'\s+', header=None)
            y_train = pd.read_csv(os.path.join(base_path, 'train', 'y_train.txt'),
                                  header=None).squeeze()

            X_test = pd.read_csv(test_path, sep=r'\s+', header=None)
            y_test = pd.read_csv(os.path.join(base_path, 'test', 'y_test.txt'),
                                 header=None).squeeze()

            features_path = os.path.join(base_path, 'features.txt')
            if os.path.exists(features_path):
                features = pd.read_csv(features_path, sep=r'\s+', header=None)
                feature_names = features[1].tolist()
                X_train.columns = feature_names
                X_test.columns = feature_names
                print(f"Loaded {len(feature_names)} feature names")
            else:
                feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
                X_train.columns = feature_names
                X_test.columns = feature_names
                print("Using default feature names")

            labels_path = os.path.join(base_path, 'activity_labels.txt')
            if os.path.exists(labels_path):
                activity_df = pd.read_csv(labels_path, sep=r'\s+', header=None)
                for _, row in activity_df.iterrows():
                    self.activity_labels[row[0]] = row[1]

            print(f"Training set: {X_train.shape}")
            print(f"Test set: {X_test.shape}")
            print(f"Activity labels: {self.activity_labels}")

            return X_train, X_test, y_train, y_test, feature_names

        except Exception as e:
            print(f"Error loading dataset: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None, None

    def clean_data_consistent(self, X_train, X_test, y_train, y_test):
        print("Starting consistent data cleaning...")
        print(f"Original training shape: {X_train.shape}")
        print(f"Original test shape: {X_test.shape}")

        X_combined = pd.concat([X_train, X_test], axis=0)
        X_combined_clean = X_combined.drop_duplicates()

        constant_features = X_combined_clean.columns[X_combined_clean.nunique() == 1]
        if len(constant_features) > 0:
            print(f"Removing {len(constant_features)} constant features")
            X_combined_clean = X_combined_clean.drop(columns=constant_features)

        duplicate_features = X_combined_clean.columns[X_combined_clean.columns.duplicated()]
        if len(duplicate_features) > 0:
            print(f"Removing {len(duplicate_features)} duplicate features")
            X_combined_clean = X_combined_clean.loc[:, ~X_combined_clean.columns.duplicated()]

        print("Checking for highly correlated features...")
        corr_matrix = X_combined_clean.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]

        if len(high_corr_features) > 0:
            print(f"Removing {len(high_corr_features)} highly correlated features")
            X_combined_clean = X_combined_clean.drop(columns=high_corr_features)

        missing_count = X_combined_clean.isnull().sum().sum()
        if missing_count > 0:
            print(f"Filling {missing_count} missing values with median")
            X_combined_clean = X_combined_clean.fillna(X_combined_clean.median())

        X_train_clean = X_combined_clean.iloc[:len(X_train)]
        X_test_clean = X_combined_clean.iloc[len(X_train):]

        y_train_clean = y_train.iloc[X_train_clean.index]
        y_test_clean = y_test.iloc[X_test_clean.index]

        print(f"Final training shape: {X_train_clean.shape}")
        print(f"Final test shape: {X_test_clean.shape}")
        print(f"Common features: {X_train_clean.shape[1]}")

        self.common_features = X_train_clean.columns.tolist()

        return X_train_clean, X_test_clean, y_train_clean, y_test_clean

    def normalize_data(self, data, method='standard'):
        if method == 'standard':
            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(data)
        elif method == 'minmax':
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(data)
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
        return normalized_data, scaler

    def apply_pca(self, X, n_components=0.95):
        print(f"Applying PCA to reduce dimensions from {X.shape[1]}...")
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X)
        print(f"Reduced to {X_pca.shape[1]} components")
        print(f"Explained variance ratio: {np.sum(self.pca.explained_variance_ratio_):.3f}")

        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance')
        plt.grid(True)
        plt.savefig('outputs/pca_explained_variance.png', dpi=150, bbox_inches='tight')
        plt.show()

        return X_pca

    def full_preprocessing_pipeline(self, data_path):
        print("Starting UCI HAR Dataset preprocessing pipeline...")

        X_train_raw, X_test_raw, y_train_raw, y_test_raw, feature_names = self.load_uci_har_dataset(data_path)

        if X_train_raw is None:
            print("Failed to load dataset. Please check the file structure.")
            return None, None, None, None

        print("\nCleaning data consistently across train and test...")
        X_train_clean, X_test_clean, y_train_clean, y_test_clean = self.clean_data_consistent(
            X_train_raw, X_test_raw, y_train_raw, y_test_raw
        )

        print("\nNormalizing data...")
        X_train_normalized, self.scaler = self.normalize_data(X_train_clean, method='standard')
        X_test_normalized = self.scaler.transform(X_test_clean)

        print("\nApplying PCA...")
        X_train_pca = self.apply_pca(X_train_normalized)
        X_test_pca = self.pca.transform(X_test_normalized)

        print("\nPreprocessing completed successfully!")
        print(f"Final training set: {X_train_pca.shape}")
        print(f"Final test set: {X_test_pca.shape}")

        return X_train_pca, X_test_pca, y_train_clean, y_test_clean


def plot_dataset_overview(X_train, X_test, y_train, y_test, activity_labels):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].hist(X_train.flatten(), bins=50, alpha=0.7, label='Train', color='blue', density=True)
    axes[0, 0].hist(X_test.flatten(), bins=50, alpha=0.7, label='Test', color='red', density=True)
    axes[0, 0].set_title('Feature Distribution (Train vs Test)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    train_counts = pd.Series(y_train).value_counts().sort_index()
    activity_names = [activity_labels.get(i, f'Activity {i}') for i in train_counts.index]
    axes[0, 1].bar(activity_names, train_counts.values, color='skyblue')
    axes[0, 1].set_title('Training Set Activity Distribution')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True)

    test_counts = pd.Series(y_test).value_counts().sort_index()
    activity_names = [activity_labels.get(i, f'Activity {i}') for i in test_counts.index]
    axes[0, 2].bar(activity_names, test_counts.values, color='lightcoral')
    axes[0, 2].set_title('Test Set Activity Distribution')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].grid(True)

    axes[1, 0].plot(X_train[:100, 0], label='Train', alpha=0.7)
    axes[1, 0].plot(X_test[:100, 0], label='Test', alpha=0.7)
    axes[1, 0].set_title('First Feature (First 100 samples)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    if X_train.shape[1] >= 20:
        corr_matrix = np.corrcoef(X_train[:, :20].T)
        im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[1, 1].set_title('Feature Correlation Matrix (First 20 features)')
        plt.colorbar(im, ax=axes[1, 1])
    else:
        corr_matrix = np.corrcoef(X_train.T)
        im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[1, 1].set_title('Feature Correlation Matrix')
        plt.colorbar(im, ax=axes[1, 1])

    if X_train.shape[1] >= 2:
        scatter = axes[1, 2].scatter(X_train[:, 0], X_train[:, 1], c=y_train, alpha=0.6, cmap='tab10')
        axes[1, 2].set_title('PCA Components Scatter Plot')
        axes[1, 2].set_xlabel('First Principal Component')
        axes[1, 2].set_ylabel('Second Principal Component')
        plt.colorbar(scatter, ax=axes[1, 2])

    plt.tight_layout()
    plt.savefig('outputs/dataset_overview.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n Dataset Summary ")
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"Activities: {len(np.unique(y_train))} classes")
    print(f"Feature range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"Feature mean: {X_train.mean():.3f}, std: {X_train.std():.3f}")
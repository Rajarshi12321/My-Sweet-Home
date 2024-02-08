from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    dataset_path: Path
    preprocessor_path: Path
    tracked_preprocessor_path: Path


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    training_data: Path
    trained_model_file_path: Path
    trained_model_file_path_rent: Path
    mlflow_uri: str
    model_params: dict


@dataclass(frozen=True)
class DataTransformationRecommendConfig:
    root_dir: Path
    dataset_path: Path
    processed_dataset_path: Path
    recommend_dataset_path: Path
    tracked_recommend_dataset_path: Path

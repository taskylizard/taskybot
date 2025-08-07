import os
import subprocess
from pathlib import Path
from .common import (
    MODEL_NAME,
    MODEL_PATH,
    get_user_data_path,
    get_user_model_path,
    WANDB_PROJECT,
)

MINUTES = 60  # seconds
HOURS = 60 * MINUTES

REMOTE_CONFIG_PATH = Path("/llama3_1_8B_lora.yaml")


def download_model():
    subprocess.run(
        [
            "tune",
            "download",
            MODEL_NAME,
            f"--output-dir={MODEL_PATH}",
            "--ignore-patterns",
            "original/consolidated.00.pth",
        ]
    )


wandb_args = [
    "metric_logger._component_=torchtune.training.metric_logging.WandBLogger",
    f"metric_logger.project={WANDB_PROJECT}",
]


def finetune(user_id: str, cleanup: bool = True):
    """Fine-tune a model on the user's Discord messages with torchtune.

    Args:
        user_id: The Discord user ID to fine-tune on.
        cleanup: Remove user data after fine-tuning. On by default.
    """

    if not Path("/vol/model").exists():
        print("Downloading model...")
        download_model()

    data_path = get_user_data_path(user_id)

    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"No training data found for user {user_id}. Run scraping first."
        )

    output_dir = get_user_model_path(user_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Fine-tuning model for user {user_id}...")
    print(f"Data path: {data_path}")
    print(f"Output dir: {output_dir}")

    subprocess.run(
        [
            "tune",
            "run",
            "lora_finetune_single_device",
            "--config",
            REMOTE_CONFIG_PATH,
            f"output_dir={output_dir}",
            f"dataset_path={data_path}",
            f"model_path={MODEL_PATH}",
            *wandb_args,
        ]
    )

    print(f"Fine-tuning completed for user {user_id}")

    if cleanup and user_id != "test":
        # Delete scraped data after fine-tuning to save space
        print(f"Cleaning up training data for user {user_id}")
        os.remove(data_path)

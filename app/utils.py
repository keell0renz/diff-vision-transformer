from safetensors.torch import save_file, load_file  # type: ignore
from huggingface_hub import HfApi  # type: ignore
from dotenv import dotenv_values
from typing import Dict
import torch
import os


def load_environment():
    """
    Load environment variables from environment, if absent, load from .env file.

    This is a convenience function for running Docker container of this repository.
    """

    dotenv_path = ".env"
    env_vars = dotenv_values(dotenv_path)

    for key, value in env_vars.items():
        if key not in os.environ and value is not None:
            os.environ[key] = value


def save_model_to_safetensors(model: torch.nn.Module, path: str) -> None:
    """
    Save a PyTorch model's state dict to a file using safetensors format.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        path (str): The path to save the model to.

    Returns:
        None
    """

    state_dict = model.state_dict()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(state_dict, path)


def load_model_from_safetensors(
    path: str, device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Load a PyTorch model's state dict from a safetensors file.

    Args:
        path (str): The path to the safetensors file.
        device (str): The device to load the tensors onto. Defaults to CPU.

    Returns:
        Dict[str, torch.Tensor]: The state dict of the PyTorch model.
    """

    if not os.path.isfile(path):
        raise ValueError(f"{path} is not a valid file.")

    state_dict = load_file(path, device=device)
    return state_dict


def upload_file_to_hf(filename_in: str, directory_out: str) -> None:
    """
    Upload a file to the Hugging Face repository.
    """

    if not os.path.isfile(filename_in):
        raise ValueError(f"{filename_in} is not a valid file.")

    try:
        repo_id = os.environ["HF_REPOSITORY"]
    except KeyError:
        raise ValueError("HF_REPOSITORY environment variable is not set.")

    api = HfApi()

    filename_out = os.path.join(directory_out, os.path.basename(filename_in))

    api.upload_file(
        path_or_fileobj=filename_in,
        path_in_repo=filename_out,
        commit_message=f"Added {os.path.basename(filename_in)} file.",
        repo_id=repo_id,
    )


def download_file_from_hf(filename_in: str, directory_out: str) -> None:
    """
    Download a file from the Hugging Face repository.
    """

    if not os.path.isdir(directory_out):
        raise ValueError(f"{directory_out} is not a valid directory.")

    try:
        repo_id = os.environ["HF_REPOSITORY"]
    except KeyError:
        raise ValueError("HF_REPOSITORY environment variable is not set.")

    api = HfApi()

    if filename_in.startswith('/'):
        filename_in = filename_in[1:]

    api.hf_hub_download(
        repo_id=repo_id,
        filename=filename_in,
        local_dir=directory_out,
    )

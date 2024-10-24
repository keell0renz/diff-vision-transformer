from app.utils import (
    upload_file_to_hf,
    download_file_from_hf,
    load_environment,
    pytorch_health_check,
)
from typer import Typer, Argument


app = Typer()


@app.command()
def upload(
    in_file: str = Argument(..., help="Path to the file to upload."),
    out_dir: str = Argument(..., help="Path to the directory to upload the file to."),
):
    """
    Upload a file to the Hugging Face repository.
    """

    upload_file_to_hf(in_file, out_dir)


@app.command()
def download(
    in_file: str = Argument(..., help="Path to the file to download."),
    out_dir: str = Argument(..., help="Path to the directory to download the file to."),
):
    """
    Download a file from the Hugging Face repository.
    """

    download_file_from_hf(in_file, out_dir)


@app.command()
def health():
    """
    Perform a health check for PyTorch and print relevant information.
    """
    pytorch_health_check()


if __name__ == "__main__":
    load_environment()
    app()

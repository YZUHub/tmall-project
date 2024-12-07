from typer import Typer

from utils import check_metrics
from workflow import train_model, validate_model

app = Typer()


@app.command(name="train")
def train():
    """Train the model."""
    print("Training the model.")
    metrics = model_trainer()
    print("## Model Verification Report")
    print(check_metrics(metrics))


@app.command(name="verify")
def verify():
    """Verify project submissions."""
    metrics = model_validator()
    print("## Model Verification Report")
    print(check_metrics(metrics))


if __name__ == "__main__":
    app()
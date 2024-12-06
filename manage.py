#!/usr/bin/env python
"""Command-line utility using Typer for the tmall-project."""
import typer

app = typer.Typer()

@app.command()
def run():
    """Run the main script."""
    typer.echo("Running the main script...")

@app.command()
def hello(name: str):
    """Say hello to a user."""
    typer.echo(f"Hello, {name}!")

@app.command()
def version():
    """Display the project version."""
    typer.echo("tmall-project version 1.0.0")

if __name__ == "__main__":
    app()

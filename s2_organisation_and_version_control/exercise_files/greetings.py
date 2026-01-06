import typer

def main(count: int = 1) -> None:
    """Print greeting messages."""
    for _ in range(count):
        print("Hello world!")


if __name__ == "__main__":
    typer.run(main)
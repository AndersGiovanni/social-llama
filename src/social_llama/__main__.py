"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Social Llama."""


if __name__ == "__main__":
    main(prog_name="social-llama")  # pragma: no cover

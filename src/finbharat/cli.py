import click
from pathlib import Path


@click.group()
def cli():
    pass


@cli.command()
@click.option("--data-root", type=Path, default=Path("dataset"), help="Path to dataset directory")
@click.option("--output-dir", type=Path, default=Path("results"), help="Output directory")
@click.option("--models", multiple=True, default=("qwen3-8b", "llama3.1-8b"), help="Model keys to evaluate")
@click.option("--difficulty", type=click.Choice(["easy", "medium", "hard", "multihop"]), default="easy")
@click.option("--max-per-company", type=int, default=None, help="Max QA pairs per company")
@click.option("--api-key", type=str, default=None, envvar="NVIDIA_API_KEY", help="NIM API key")
def evaluate(data_root, output_dir, models, difficulty, max_per_company, api_key):
    from finbharat.eval.evaluate import run_evaluation
    run_evaluation(
        data_root=data_root,
        output_dir=output_dir,
        model_keys=list(models),
        difficulty=difficulty,
        max_per_company=max_per_company,
        api_key=api_key,
    )


@cli.command()
@click.option("--data-root", type=Path, default=Path("dataset"), help="Path to dataset directory")
@click.option("--difficulty", type=click.Choice(["easy", "medium", "hard", "multihop"]), default="easy")
def sample(data_root, difficulty):
    from finbharat.data.loader import FinBharatDataset, SAMPLE_COMPANIES
    from collections import Counter
    dataset = FinBharatDataset(data_root)
    records = dataset.load_sample(difficulty=difficulty)
    qtypes = Counter(r.question_type for r in records)
    companies = Counter(r.company for r in records)
    click.echo(f"Loaded {len(records)} QA records for difficulty={difficulty}")
    click.echo(f"Companies: {dict(companies)}")
    click.echo(f"Question types: {dict(qtypes)}")
    if records:
        click.echo(f"\nSample question:")
        click.echo(f"  Q: {records[0].question}")
        click.echo(f"  A: {records[0].answer}")
        click.echo(f"  Type: {records[0].question_type}")


@cli.command()
def models():
    from finbharat.models.runner import PREDEFINED_MODELS
    for key, config in PREDEFINED_MODELS.items():
        click.echo(f"  {key}: {config.name} (model_id={config.model_id})")


if __name__ == "__main__":
    cli()

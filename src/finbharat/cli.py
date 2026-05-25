import click
from pathlib import Path


@click.group()
def cli():
    pass


@cli.command()
@click.option("--data-root", type=Path, default=Path("dataset"), help="Path to dataset directory")
@click.option("--output-dir", type=Path, default=Path("results"), help="Output directory")
@click.option("--models", multiple=True, default=("llama3.1-8b",), help="Model keys (see: finbharat models)")
@click.option("--difficulty", type=click.Choice(["easy", "medium", "hard", "multihop"]), default="easy")
@click.option("--regime", type=click.Choice(["zero_shot", "few_shot", "closed_book", "few_shot_closed"]),
              default="zero_shot", help="Evaluation regime")
@click.option("--table-format", type=click.Choice(["html", "markdown", "linearized"]),
              default="html", help="How to serialize HTML tables in context")
@click.option("--all-companies", is_flag=True, default=False, help="Run on all 187 companies")
@click.option("--max-per-company", type=int, default=None, help="Cap QA pairs per company")
@click.option("--api-key", type=str, default=None, envvar="NVIDIA_API_KEY", help="Override NIM API key")
@click.option("--llm-judge", "llm_judge_model", type=str, default=None,
              help="Model key to use as LLM numerical judge (e.g. llama3.1-8b). Omit to skip.")
@click.option("--vllm-model", type=str, default=None,
              help="Model ID served by a local vLLM server (e.g. 'fingpt' or 'meta-llama/Meta-Llama-3-8B'). "
                   "Overrides --models when set.")
@click.option("--vllm-host", type=str, default="localhost",
              help="Host where vLLM is running (default: localhost). Use SSH tunnel or direct IP.")
@click.option("--vllm-port", type=int, default=8000,
              help="Port where vLLM server listens (default: 8000).")
@click.option("--vllm-completion", is_flag=True, default=False,
              help="Use /completions endpoint with Alpaca format (for FinMA and similar instruction-tuned models).")
@click.option("--vllm-max-context", type=int, default=0,
              help="Truncate context to N chars before sending (e.g. 3000 for FinMA with 2048-token limit). 0=no limit.")
def evaluate(data_root, output_dir, models, difficulty, regime, table_format,
             all_companies, max_per_company, api_key, llm_judge_model,
             vllm_model, vllm_host, vllm_port, vllm_completion, vllm_max_context):
    from finbharat.eval.evaluate import run_evaluation
    from finbharat.models.runner import make_vllm_config, PREDEFINED_MODELS

    if vllm_model:
        # Register the local vLLM model dynamically and use it
        cfg = make_vllm_config(vllm_model, host=vllm_host, port=vllm_port,
                               use_completion=vllm_completion,
                               max_context_chars=vllm_max_context)
        PREDEFINED_MODELS["__vllm__"] = cfg
        model_keys = ["__vllm__"]
        click.echo(f"Using local vLLM model: {vllm_model} at {cfg.api_base}")
    else:
        model_keys = list(models)

    run_evaluation(
        data_root=data_root,
        output_dir=output_dir,
        model_keys=model_keys,
        difficulty=difficulty,
        regime=regime,
        table_format=table_format,
        all_companies=all_companies,
        max_per_company=max_per_company,
        api_key=api_key,
        llm_judge_model=llm_judge_model,
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
@click.option("--results-dir", type=Path, default=Path("results"))
@click.option("--detailed", is_flag=True)
@click.option("--compare", is_flag=True)
@click.option("--errors", is_flag=True)
@click.option("--failures", is_flag=True)
@click.option("--all", "show_all", is_flag=True)
def analyze(results_dir, detailed, compare, errors, failures, show_all):
    """Analyze and print results summary tables."""
    from finbharat.analysis.results_analyzer import (
        load_aggregates, print_summary_table, print_by_difficulty,
        print_by_question_type, print_regime_comparison,
        print_error_summary, print_top_failures,
    )
    records = load_aggregates(results_dir)
    if not records:
        click.echo(f"No results found in {results_dir}/aggregates")
        return
    click.echo(f"\n{'='*100}")
    click.echo(f"  FinBharat Results  |  {len(records)} run(s)  |  {results_dir}")
    click.echo(f"{'='*100}\n")
    print_summary_table(records)
    if len({r["_difficulty"] for r in records}) > 1 or show_all:
        click.echo(f"\n{'='*100}")
        click.echo("  By Difficulty Tier")
        click.echo(f"{'='*100}")
        print_by_difficulty(records)
    if detailed or show_all:
        click.echo(f"\n{'='*100}")
        click.echo("  By Question Type")
        click.echo(f"{'='*100}")
        print_by_question_type(records)
    if compare or show_all:
        click.echo(f"\n{'='*100}")
        click.echo("  Regime Comparison")
        click.echo(f"{'='*100}")
        print_regime_comparison(records)
    if errors or show_all:
        click.echo(f"\n{'='*100}")
        click.echo("  API Error Summary")
        click.echo(f"{'='*100}")
        print_error_summary(results_dir)
    if failures or show_all:
        click.echo(f"\n{'='*100}")
        click.echo("  Near-Miss Failures")
        click.echo(f"{'='*100}")
        print_top_failures(results_dir)


@cli.command()
def models():
    from finbharat.models.runner import PREDEFINED_MODELS
    for key, config in PREDEFINED_MODELS.items():
        click.echo(f"  {key}: {config.name} (model_id={config.model_id})")


if __name__ == "__main__":
    cli()

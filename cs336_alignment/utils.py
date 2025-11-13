from pathlib import Path

def load_prompt_template(prompt_name: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_path = Path("cs336_alignment") / "prompts" / f"{prompt_name}.prompt"
    with open(prompt_path) as f:
        return f.read()

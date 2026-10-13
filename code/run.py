import sys
import subprocess
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()
HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent  # Project root folder

STEPS = [
    ("Download and load depth model", "load_model.py"),
    ("Joint SfM + metric depth + per-video fusion", "reconstruct.py"),
    ("Occlusion-aware change detection + labeling", "detect.py"),
    ("Rerun visualization", "visualize.py"),
]

def run_pipeline():
    console.print(Panel.fit("[bold green]3D Room Change Detection[/bold green]", border_style="cyan"))
    
    for idx, (name, script) in enumerate(STEPS, start=1):
        path = HERE / script
        if not path.exists():
            console.print(f"[bold red]✖ Missing script:[/bold red] {path}")
            sys.exit(1)
            
        console.print(f"\n[bold yellow]──── Stage {idx}/{len(STEPS)}: {name} ({script}) ────[/bold yellow]")
        result = subprocess.run([sys.executable, str(path)], cwd=HERE)
        
        if result.returncode != 0:
            console.print(f"\n[bold white on red] ✖ FAILED AT STAGE {idx}: {name} [/bold white on red]")
            sys.exit(result.returncode)

    console.print("\n[bold white on green] ✨ DONE ✨ [/bold white on green] Opening Rerun...")
    
    rrd_file = ROOT / "output" / "scene_changes.rrd"
    subprocess.run([sys.executable, "-m", "rerun", str(rrd_file)])

if __name__ == "__main__":
    run_pipeline()
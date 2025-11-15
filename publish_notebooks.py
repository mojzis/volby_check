#!/usr/bin/env python3
"""
Export Marimo notebooks to HTML with error handling and manifest generation.
"""

import subprocess
import json
import sys
from pathlib import Path
from typing import Dict, List

# Configuration
NOTEBOOKS = [
    {
        "file": "suspicious_zeros_analysis.py",
        "title": "Basic Probability Analysis",
        "description": "Core statistical analysis of zero-vote cases using binomial probability",
        "recommended": False
    },
    {
        "file": "suspicious_zeros_with_obec_analysis.py",
        "title": "Enhanced Analysis with Municipality Context",
        "description": "Complete analysis with municipality size adjustments and geographic context",
        "recommended": False
    },
    {
        "file": "suspicious_zeros_obec_specific.py",
        "title": "Municipality-Specific Analysis",
        "description": "Detailed analysis with municipality-level context and size adjustments - the most comprehensive view",
        "recommended": True
    },
    {
        "file": "suspicious_cases_analysis.py",
        "title": "List of Suspicious Cases",
        "description": "Extended analysis of suspicious voting patterns",
        "recommended": False
    },
    {
        "file": "advanced_anomaly_detection.py",
        "title": "Advanced Anomaly Detection",
        "description": "Multi-technique validation using machine learning (Isolation Forest), multi-party analysis, and composite scoring",
        "recommended": True
    }
]

OUTPUT_DIR = Path("html_output")
MANIFEST_FILE = OUTPUT_DIR / "manifest.json"


def export_notebook(notebook: Dict[str, str], output_dir: Path) -> bool:
    """
    Export a single Marimo notebook to HTML.

    Args:
        notebook: Dictionary with notebook configuration
        output_dir: Directory to save HTML output

    Returns:
        True if export succeeded, False otherwise
    """
    notebook_file = notebook["file"]
    output_file = output_dir / f"{Path(notebook_file).stem}.html"

    print(f"\n{'='*60}")
    print(f"Exporting {notebook_file}...")
    print(f"{'='*60}")

    # Check if notebook file exists
    if not Path(notebook_file).exists():
        print(f"❌ SKIPPED: File {notebook_file} does not exist")
        return False

    try:
        # Run marimo export command
        result = subprocess.run(
            ["uv", "run", "marimo", "export", "html", "--no-include-code", notebook_file, "-o", str(output_file)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            print(f"✅ SUCCESS: Exported to {output_file}")
            return True
        else:
            print(f"❌ FAILED: Export failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"❌ FAILED: Export timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ FAILED: Unexpected error: {e}")
        return False


def create_manifest(successful_notebooks: List[Dict], output_dir: Path) -> None:
    """
    Create a JSON manifest of successfully published notebooks.

    Args:
        successful_notebooks: List of notebook configurations that were successfully exported
        output_dir: Directory where manifest will be saved
    """
    manifest = {
        "notebooks": successful_notebooks,
        "total_count": len(successful_notebooks),
        "generated_at": "{{ GITHUB_RUN_ID }}"  # Will be replaced by GitHub Actions
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ Manifest created: {manifest_path}")
    print(f"   Successfully published: {len(successful_notebooks)} notebooks")


def create_index_html(successful_notebooks: List[Dict], output_dir: Path) -> None:
    """
    Create an index.html page that lists all successfully published notebooks.

    Args:
        successful_notebooks: List of notebook configurations that were successfully exported
        output_dir: Directory where index.html will be saved
    """
    # Sort notebooks: recommended first, then by title
    sorted_notebooks = sorted(
        successful_notebooks,
        key=lambda x: (not x.get('recommended', False), x['title'])
    )

    # Generate notebook items HTML
    notebook_items = []
    for nb in sorted_notebooks:
        html_file = f"{Path(nb['file']).stem}.html"
        recommended_class = " recommended" if nb.get('recommended', False) else ""

        item = f"""
                <div class="notebook-item{recommended_class}">
                    <a href="{html_file}">{nb['title']}</a>
                    <p>{nb['description']}</p>
                </div>"""
        notebook_items.append(item)

    notebooks_html = "\n".join(notebook_items)

    # Create full HTML page
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Czech Election Analysis - Notebooks</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
            background: #f5f5f5;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .notebook-list {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .notebook-item {{
            margin: 15px 0;
            padding: 15px;
            border-left: 4px solid #3498db;
            background: #f8f9fa;
            border-radius: 4px;
        }}
        .notebook-item a {{
            color: #2980b9;
            text-decoration: none;
            font-weight: bold;
            font-size: 1.1em;
        }}
        .notebook-item a:hover {{
            color: #3498db;
            text-decoration: underline;
        }}
        .notebook-item p {{
            margin: 5px 0 0 0;
            color: #666;
        }}
        .recommended {{
            border-left-color: #27ae60;
        }}
        .recommended::before {{
            content: "⭐ RECOMMENDED";
            display: block;
            color: #27ae60;
            font-size: 0.8em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .status-info {{
            background: #e8f4f8;
            border: 1px solid #bee5eb;
            border-radius: 4px;
            padding: 15px;
            margin: 20px 0;
        }}
        .status-info p {{
            margin: 5px 0;
            color: #0c5460;
        }}
        footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>Czech Election Analysis - Interactive Notebooks</h1>

    <p>Statistical analysis of suspicious zero-vote cases in Czech parliamentary elections using probability theory and geographic context.</p>

    <div class="status-info">
        <p><strong>Published Notebooks:</strong> {len(successful_notebooks)} of {len(NOTEBOOKS)} available</p>
    </div>

    <h2>Available Notebooks</h2>

    <div class="notebook-list">
{notebooks_html}
    </div>

    <h2>About This Analysis</h2>
    <p>This project identifies statistically improbable zero-vote cases where major political parties received no votes in specific voting commissions. The analysis uses:</p>
    <ul>
        <li><strong>Binomial Probability:</strong> P(0 votes) = (1-p)^n</li>
        <li><strong>Municipality Size Context:</strong> Adjusts for urban vs rural voting patterns</li>
        <li><strong>Thresholds:</strong> P &lt; 0.1% = highly suspicious, P &lt; 1% = suspicious</li>
        <li><strong>Machine Learning:</strong> Advanced anomaly detection using multiple validation techniques</li>
    </ul>

    <footer>
        <p>Data source: Czech Statistical Office (CZSO) - <a href="https://www.volby.cz/opendata/">volby.cz/opendata</a></p>
        <p>Generated with Marimo interactive notebooks</p>
    </footer>
</body>
</html>
"""

    index_path = output_dir / "index.html"
    with open(index_path, 'w') as f:
        f.write(html_content)

    print(f"✅ Index page created: {index_path}")


def main():
    """Main export workflow."""
    print("="*60)
    print("Marimo Notebook Export with Error Handling")
    print("="*60)

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"\n📁 Output directory: {OUTPUT_DIR}")

    # Export notebooks with error handling
    successful_notebooks = []
    failed_notebooks = []

    for notebook in NOTEBOOKS:
        success = export_notebook(notebook, OUTPUT_DIR)

        if success:
            successful_notebooks.append(notebook)
        else:
            failed_notebooks.append(notebook)

    # Print summary
    print("\n" + "="*60)
    print("EXPORT SUMMARY")
    print("="*60)
    print(f"✅ Successful: {len(successful_notebooks)}")
    print(f"❌ Failed: {len(failed_notebooks)}")

    if failed_notebooks:
        print("\nFailed notebooks:")
        for nb in failed_notebooks:
            print(f"  - {nb['file']}")

    # Create manifest and index
    if successful_notebooks:
        create_manifest(successful_notebooks, OUTPUT_DIR)
        create_index_html(successful_notebooks, OUTPUT_DIR)
        print("\n✅ Publishing complete!")
        return 0
    else:
        print("\n❌ No notebooks were successfully exported!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

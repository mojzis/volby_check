# CLAUDE.md - Repository Guide

## Project Overview

This is a Czech election analysis project that identifies statistically suspicious zero-vote cases in parliamentary elections using probability theory and geographic context.

## Repository Structure

### Main Analysis Files

- **`suspicious_zeros_analysis.py`** - Basic Marimo notebook with probability analysis
- **`suspicious_zeros_with_obec_analysis.py`** - Enhanced analysis with municipality size context (RECOMMENDED)
- **`suspicious_zeros_obec_specific.py`** - Municipality-specific analysis
- **`suspicious_cases_analysis.py`** - Additional suspicious cases analysis
- **`election_analysis.py`** - Core election data analysis utilities
- **`show_most_suspicious.py`** - Script to display most suspicious cases

### Support Files

- **`main.py`** - Entry point
- **`README.md`** - User-facing documentation
- **`FINDINGS_GUIDE.md`** - Guide for interpreting findings
- **`pyproject.toml`** - Python dependencies (uv package manager)

## Technology Stack

- **Python**: 3.11+
- **Package Manager**: uv
- **Interactive Notebooks**: Marimo (NOT Jupyter)
- **Key Libraries**: pandas, plotly, scipy, numpy, requests, pyarrow
- **Data Format**: Parquet files (cached locally as `election_data.parquet`)

## Key Concepts

### Statistical Method
The analysis uses binomial probability: **P(0 votes) = (1-p)^n**
- `p` = party's vote probability
- `n` = total votes in commission
- Thresholds: P < 0.1% = highly suspicious, P < 1% = suspicious

### Municipality Size Context
The enhanced analysis categorizes municipalities into 5 size tiers and uses size-adjusted probabilities to account for urban vs rural party preferences.

## Working with This Repository

### Running Analysis
```bash
# Install dependencies
uv sync

# Run main analysis (opens in browser)
marimo edit suspicious_zeros_with_obec_analysis.py
```

### Data Source
- Czech Statistical Office (CZSO): https://www.volby.cz/opendata/
- Data is downloaded and cached locally on first run

### File Naming
- **"volby"** = elections (Czech)
- **"obec"** = municipality (Czech)
- **"suspicious zeros"** = commissions where major parties got 0 votes

## Important Notes for Claude

1. **Marimo vs Jupyter**: This project uses Marimo notebooks (.py files), not Jupyter (.ipynb). They are edited with `marimo edit` command.

2. **Czech Context**: Some variable names and concepts are in Czech. Common terms:
   - volby = elections
   - obec = municipality
   - okres = district
   - okrsek = commission/precinct

3. **Statistical Focus**: The core value is probability calculations. Always preserve statistical accuracy when modifying analysis code.

4. **Interactive Visualizations**: Uses Plotly for interactive charts. Maintain interactivity when adding new visualizations.

5. **Data Caching**: Election data is cached locally. Don't repeatedly download if parquet file exists.

## Common Tasks

- **Add new analysis**: Create new Marimo notebook or extend existing ones
- **Modify probability thresholds**: Look for hardcoded values like 0.01 (1%), 0.001 (0.1%)
- **Add visualizations**: Use Plotly, ensure charts are interactive
- **Update data source**: Modify download logic in analysis files
- **Change municipality categories**: Update size tier logic in obec-specific files

## Testing and Pre-Push Requirements

**CRITICAL: Always test notebooks before committing and pushing changes!**

### Required Testing Steps

Before pushing any notebook changes, you MUST:

1. **Test the notebook locally** by running the export command:
   ```bash
   uv run marimo export html --no-include-code <notebook_name>.py -o test_output.html
   ```

2. **Verify the export succeeds** without errors

3. **Check the generated HTML** opens correctly in a browser

4. **Validate data integrity**:
   - Data downloads successfully
   - Probability calculations are accurate
   - Visualizations render correctly in browser
   - No errors in Marimo execution

### Common Testing Patterns

```bash
# Test a single notebook
uv run marimo export html --no-include-code advanced_anomaly_detection.py -o test.html

# Test all notebooks (using the publishing script)
python publish_notebooks.py

# Run notebook interactively to debug
marimo edit advanced_anomaly_detection.py
```

### Why This Matters

- GitHub Pages publishes from notebooks, so broken exports mean broken public pages
- The publishing workflow now has error handling, but prevention is better than recovery
- Testing locally catches column mismatches, missing data, and other runtime errors

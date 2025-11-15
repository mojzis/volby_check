# Czech Election Zero-Vote Analysis

Statistical analysis to identify suspicious zero-vote cases in Czech parliamentary elections using probability theory and geographic context.

## Overview

This project provides interactive Marimo notebooks that analyze election data to find statistically improbable cases where major parties received zero votes in specific commissions. The analysis uses binomial probability to quantify how unlikely each zero-vote case is.

## Files

### 1. `suspicious_zeros_analysis.py`
Basic probability analysis focusing on the top 3 largest zero-vote commissions per party.

**Features:**
- Calculates binomial probability: P(0 votes) = (1-p)^n
- Identifies cases with P < 1% as suspicious
- Visualizations: heatmaps, bubble charts, suspiciousness scores
- Simple and fast

**Run with:**
```bash
marimo edit suspicious_zeros_analysis.py
```

### 2. `suspicious_zeros_with_obec_analysis.py` (Recommended)
Enhanced analysis that adds geographic and demographic context using municipality (OBEC) size.

**Features:**
- Categorizes municipalities into 5 size tiers
- Calculates party performance by municipality size
- Uses **size-adjusted probabilities** that account for urban vs rural preferences
- More accurate detection of suspicious cases
- Additional visualizations showing geographic patterns

**Run with:**
```bash
marimo edit suspicious_zeros_with_obec_analysis.py
```

## How It Works

### Statistical Method

For each commission where a top party received 0 votes, we calculate the probability this would happen by chance:

**P(0 votes) = (1 - p)^n**

Where:
- `p` = party's vote probability (either overall or size-adjusted)
- `n` = total votes cast in the commission

### Example

If a party has 20% national support and a 500-vote commission gives them 0 votes:
- Expected votes: ~100
- P(0 votes) = (0.8)^500 H 10^-50
- **Verdict**: Essentially impossible, highly suspicious!

### Suspiciousness Thresholds

- **P < 0.1%** (0.001): Highly suspicious, essentially impossible by chance
- **P < 1%** (0.01): Suspicious, warrants investigation
- **P < 10%**: Unlikely but possible
- **P > 10%**: Could reasonably happen by chance

## Why Municipality Size Matters

The enhanced analysis (file #2) is more accurate because:

1. **Urban vs Rural Parties**: Some parties are stronger in cities, others in small towns
2. **Context-Aware**: A zero for an urban party in a large city is MORE suspicious than in a rural area
3. **Size-Adjusted Probability**: Uses `p_category` instead of overall `p` for more accurate estimates

## Key Visualizations

### 1. Probability Heatmap
Shows -logÅÄ(P) for top 3 cases per party. Higher values = more suspicious.

### 2. Bubble Chart
Commission size vs probability. Red points = suspicious (P < 1%).

### 3. Party Performance by Municipality Size
Identifies which parties are urban-friendly vs rural-friendly.

### 4. Geographic Distribution
Shows if suspicious zeros cluster in specific municipality sizes.

### 5. Suspiciousness Score
Overall ranking showing which parties have the most statistically improbable zero-vote cases.

## Installation

```bash
# Install dependencies
uv sync

# Or with pip
pip install marimo pandas plotly scipy numpy requests
```

## Usage

1. Run a Marimo notebook:
   ```bash
   marimo edit suspicious_zeros_with_obec_analysis.py
   ```

2. The notebook will:
   - Download election data from volby.cz
   - Cache it locally as `election_data.parquet`
   - Perform probability analysis
   - Generate interactive visualizations

3. Explore the interactive charts in your browser

## What to Look For

### Red Flags

1. **Very low probabilities** (P < 0.1%): Nearly impossible by chance
2. **Large commission zeros**: More suspicious than small commission zeros
3. **Wrong context zeros**: Urban party getting 0 in cities, or rural party getting 0 in villages
4. **Clustered patterns**: Multiple suspicious zeros in the same geographic area or size category

### Interpretation

- **Individual anomalies**: Could be data entry errors or local factors
- **Systematic patterns**: More concerning, suggests potential counting issues
- **Context matters**: Always consider the party's typical performance in that type of area

## Data Source

Czech Statistical Office (CZSO): https://www.volby.cz/opendata/

Data files cached locally after first download for faster subsequent runs.

## Technical Details

- **Language**: Python 3.11+
- **Framework**: Marimo (interactive notebooks)
- **Key Libraries**: pandas, plotly, scipy, numpy
- **Statistical Method**: Binomial distribution
- **Visualization**: Interactive Plotly charts

## Contributing

The analysis can be extended with:
- Regional breakdowns (beyond municipality size)
- Temporal patterns (if multiple elections are analyzed)
- Comparison with historical election data
- Machine learning for pattern detection

## License

This is an analysis tool for public election data. Use responsibly for legitimate election monitoring purposes.

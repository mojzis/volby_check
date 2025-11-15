# How to Identify Most Suspicious Cases

## Updated Notebooks (Educational Content Removed)

All notebooks have been cleaned up to focus on data-driven findings only:

1. **suspicious_zeros_with_obec_analysis.py** - Main analysis with three-tier probability system
2. **suspicious_zeros_obec_specific.py** - OBEC-specific focused analysis
3. **show_most_suspicious.py** - Command-line script to list top suspicious cases

## Running the Analysis

### Option 1: Marimo Notebooks (Interactive)
```bash
marimo edit suspicious_zeros_obec_specific.py
```

### Option 2: Command-line Script
```bash
python show_most_suspicious.py
```

## What to Look For in the Results

The analysis calculates three probability levels:

1. **Overall** - National party percentage
2. **Size-Category** - Adjusted for municipality size
3. **OBEC-Specific** - Adjusted for specific municipality (most accurate)

### Key Columns in Results

- **Party** - Party code
- **Party_Percentage** - National vote share (%)
- **OBEC** - Municipality code
- **Commission_ID** - Specific voting commission
- **Total_Votes_In_Commission** - Size of commission
- **Vote_Share_In_OBEC** - Party's % in this specific municipality
- **Probability_of_Zero_OBEC** - Probability of getting 0 votes (OBEC-adjusted)

### Suspiciousness Flags

- **Is_Highly_Suspicious** - P(0 votes) < 0.1% - Nearly impossible by chance
- **Is_Suspicious** - P(0 votes) < 1% - Very unlikely

### What Makes a Finding Suspicious

✅ **SUSPICIOUS:**
- Party gets 0 votes in a commission
- Party typically gets 5-15%+ in that municipality
- OBEC-adjusted probability < 1%

❌ **NOT SUSPICIOUS:**
- Party gets 0 votes in a commission
- Party gets <1% in that municipality overall
- Party never appeared in that municipality at all

### Bubble Chart Improvements

The bubble chart now shows:
- **Larger bubbles** = More suspicious (lower probability)
- **Minimum bubble size** = 5 (all cases visible)
- **Maximum bubble size** = 30
- **Color coding**:
  - Dark red = Highly suspicious (P < 0.1%)
  - Red = Suspicious (0.1% < P < 1%)
  - Orange = Unlikely (1% < P < 10%)
  - Light blue = Possible (P > 10%)

### Hover Data Includes

- Party name
- OBEC (municipality)
- Party's vote share in that OBEC
- Probability of zero votes
- Commission size

## Example Interpretation

```
Party: XYZ (8.5% nationally)
OBEC: Praha-1
Vote_Share_In_OBEC: 0.3%
Probability_of_Zero_OBEC: 45%
Is_Suspicious: False
```
**Interpretation:** NOT suspicious - party is very unpopular in this area

```
Party: ABC (12.2% nationally)
OBEC: Brno-střed
Vote_Share_In_OBEC: 14.5%
Probability_of_Zero_OBEC: 0.0008 (0.08%)
Is_Highly_Suspicious: True
```
**Interpretation:** HIGHLY suspicious - party normally strong in this area, getting 0 is nearly impossible

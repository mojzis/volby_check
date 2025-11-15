#!/usr/bin/env python3
"""
Quick script to show the most suspicious zero-vote cases using OBEC-specific probabilities.
"""

import pandas as pd
import requests
import zipfile
import io
from pathlib import Path

# Cache file path
parquet_file = Path("election_data.parquet")

# Load data
if parquet_file.exists():
    df = pd.read_parquet(parquet_file)
else:
    print("Downloading election data...")
    url = "https://www.volby.cz/opendata/ps2025/csv_od/pst4p.zip"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    csv_filename = zip_file.namelist()[0]

    with zip_file.open(csv_filename) as f:
        df = pd.read_csv(f)

    df.to_parquet(parquet_file)

print(f"Loaded {len(df):,} rows\n")

# Get overall party stats
party_totals = df.groupby('KSTRANA')['POC_HLASU'].sum().sort_values(ascending=False)
total_votes = party_totals.sum()
party_percentages = (party_totals / total_votes * 100).round(4)
party_probabilities = party_totals / total_votes

party_summary = pd.DataFrame({
    'Total_Votes': party_totals,
    'Percentage': party_percentages,
    'Probability': party_probabilities
})

top_parties = party_summary.head(7).index.tolist()
print(f"Top 7 parties: {', '.join(top_parties)}\n")

# Calculate party performance by OBEC
party_by_obec = df.groupby(['KSTRANA', 'OBEC'])['POC_HLASU'].agg(['sum', 'count']).reset_index()
party_by_obec.columns = ['Party', 'OBEC', 'Total_Votes', 'Num_Commissions']

obec_totals = df.groupby('OBEC')['POC_HLASU'].sum().reset_index()
obec_totals.columns = ['OBEC', 'OBEC_Total_Votes']

party_by_obec = party_by_obec.merge(obec_totals, on='OBEC')
party_by_obec['Vote_Share_In_OBEC'] = party_by_obec['Total_Votes'] / party_by_obec['OBEC_Total_Votes'] * 100
party_by_obec['Probability_In_OBEC'] = party_by_obec['Vote_Share_In_OBEC'] / 100

# Find zero-vote cases
all_commissions = df['ID_OKRSKY'].unique()
all_combinations = pd.MultiIndex.from_product(
    [top_parties, all_commissions],
    names=['Party', 'Commission_ID']
).to_frame(index=False)

actual_combinations = df[df['KSTRANA'].isin(top_parties)][['KSTRANA', 'ID_OKRSKY']].copy()
actual_combinations.columns = ['Party', 'Commission_ID']
actual_combinations['Present'] = True

combined = all_combinations.merge(actual_combinations, on=['Party', 'Commission_ID'], how='left')
zero_votes = combined[combined['Present'].isna()][['Party', 'Commission_ID']].copy()

# Get commission info
commission_info = df.groupby('ID_OKRSKY').agg({
    'POC_HLASU': 'sum',
    'OBEC': 'first'
}).reset_index()
commission_info.columns = ['Commission_ID', 'Total_Votes_In_Commission', 'OBEC']

zero_votes_df = zero_votes.merge(commission_info, on='Commission_ID')

# Merge OBEC-specific probabilities
zero_votes_df = zero_votes_df.merge(
    party_by_obec[['Party', 'OBEC', 'Vote_Share_In_OBEC', 'Probability_In_OBEC']],
    on=['Party', 'OBEC'],
    how='left'
)

# Add overall party stats
zero_votes_df['Party_Percentage'] = zero_votes_df['Party'].map(
    lambda x: party_summary.loc[x, 'Percentage']
)
zero_votes_df['Party_Probability_Overall'] = zero_votes_df['Party'].map(
    lambda x: party_summary.loc[x, 'Probability']
)

# Calculate probabilities
zero_votes_df['Probability_of_Zero_Overall'] = (
    1 - zero_votes_df['Party_Probability_Overall']
) ** zero_votes_df['Total_Votes_In_Commission']

zero_votes_df['Probability_of_Zero_OBEC'] = (
    1 - zero_votes_df['Probability_In_OBEC']
) ** zero_votes_df['Total_Votes_In_Commission']

zero_votes_df['Used_OBEC_Probability'] = zero_votes_df['Probability_In_OBEC'].notna()

zero_votes_df['Probability_of_Zero'] = zero_votes_df['Probability_of_Zero_OBEC'].fillna(
    zero_votes_df['Probability_of_Zero_Overall']
)

zero_votes_df['Is_Suspicious'] = (
    (zero_votes_df['Probability_of_Zero'] < 0.01) &
    zero_votes_df['Used_OBEC_Probability']
)

zero_votes_df['Is_Highly_Suspicious'] = (
    (zero_votes_df['Probability_of_Zero'] < 0.001) &
    zero_votes_df['Used_OBEC_Probability']
)

# Show results
print("=" * 80)
print("MOST SUSPICIOUS ZERO-VOTE CASES (OBEC-Adjusted)")
print("=" * 80)
print(f"\nTotal zero-vote cases analyzed: {len(zero_votes_df):,}")
print(f"Cases with OBEC-specific data: {zero_votes_df['Used_OBEC_Probability'].sum():,}")
print(f"Suspicious cases (P < 1%): {zero_votes_df['Is_Suspicious'].sum()}")
print(f"Highly suspicious (P < 0.1%): {zero_votes_df['Is_Highly_Suspicious'].sum()}")

print("\n" + "=" * 80)
print("TOP 30 MOST SUSPICIOUS CASES")
print("=" * 80)

most_suspicious = zero_votes_df[
    zero_votes_df['Used_OBEC_Probability']
].sort_values('Probability_of_Zero').head(30)

for idx, row in most_suspicious.iterrows():
    print(f"\n{idx+1}. Party: {row['Party']} ({row['Party_Percentage']:.1f}% nationally)")
    print(f"   OBEC: {row['OBEC']}")
    print(f"   Commission: {row['Commission_ID']}")
    print(f"   Commission size: {row['Total_Votes_In_Commission']} votes")
    print(f"   Party's share in this OBEC: {row['Vote_Share_In_OBEC']:.2f}%")
    print(f"   Probability of 0 votes (OBEC-adjusted): {row['Probability_of_Zero_OBEC']:.2e}")
    if row['Is_Highly_Suspicious']:
        print(f"   >>> HIGHLY SUSPICIOUS (P < 0.1%)")
    elif row['Is_Suspicious']:
        print(f"   >>> SUSPICIOUS (P < 1%)")

print("\n" + "=" * 80)

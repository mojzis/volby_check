#!/usr/bin/env python3
"""
Quick script to show the most suspicious zero-vote cases using OBEC-specific probabilities.
"""

import pandas as pd
import election_data_loader as edl

# Load data using shared functions
print("Loading election data...")
df = edl.load_election_data()

# Load municipality names
print("\nLoading municipality data...")
municipalities = edl.load_municipality_data()
municipality_names = dict(zip(municipalities['OBEC'], municipalities['NAZEVOBCE']))
print(f"Loaded {len(municipality_names):,} municipalities")

# Load party names
print("\nLoading party data...")
parties = edl.load_party_data()
party_names = dict(zip(parties['KSTRANA'], parties['ZKRATKAK8']))
print(f"Loaded {len(party_names):,} parties\n")

# Calculate party statistics using shared function
print("Calculating party statistics...")
party_summary, party_totals, party_percentages, party_probabilities, total_votes = edl.calculate_party_statistics(df)

# Get top 7 parties using shared function
top_parties = edl.get_top_parties(party_summary, n=7)
top_parties_names = [f"{party_names.get(p, p)} ({p})" for p in top_parties]
print(f"Top 7 parties: {', '.join(top_parties_names)}\n")

# Calculate party performance by OBEC using shared function
print("Calculating party performance by municipality...")
party_by_obec = edl.calculate_party_performance_by_obec(df)

# Find zero-vote cases using shared function
print("Finding zero-vote cases...")
zero_votes_df = edl.calculate_zero_vote_cases(df, top_parties)

# Merge OBEC probabilities and add suspiciousness flags using shared functions
print("Calculating probabilities...")
prob_analysis = edl.merge_obec_probabilities(zero_votes_df, party_by_obec, party_summary)
prob_analysis = edl.add_suspiciousness_flags(
    prob_analysis,
    suspicious_threshold=0.01,
    highly_suspicious_threshold=0.001,
    require_obec_data=True
)

# Get most suspicious cases
most_suspicious = prob_analysis[
    prob_analysis['Used_OBEC_Probability']
].sort_values('Probability_of_Zero').head(30)

# Display results
print("\n" + "="*100)
print("TOP 30 MOST SUSPICIOUS ZERO-VOTE CASES (OBEC-Specific Analysis)")
print("="*100 + "\n")

for i, row in enumerate(most_suspicious.itertuples(), 1):
    party_name = party_names.get(row.Party, row.Party)
    obec_name = municipality_names.get(row.OBEC, f"Unknown ({row.OBEC})")

    print(f"{i}. {party_name} ({row.Party})")
    print(f"   Municipality: {obec_name} (OBEC: {row.OBEC})")
    print(f"   Commission ID: {row.Commission_ID}")
    print(f"   Commission size: {row.Total_Votes_In_Commission} votes")
    print(f"   Party's share in this municipality: {row.Vote_Share_In_OBEC:.2f}%")
    print(f"   Party's national share: {row.Party_Percentage:.1f}%")
    print(f"   Probability of 0 votes: {row.Probability_of_Zero:.2e}")

    if row.Is_Highly_Suspicious:
        print(f"   Status: HIGHLY SUSPICIOUS (< 0.1% probability)")
    elif row.Is_Suspicious:
        print(f"   Status: SUSPICIOUS (< 1% probability)")
    else:
        print(f"   Status: Notable")

    print("-" * 100)

print(f"\nTotal suspicious cases (< 1%): {prob_analysis['Is_Suspicious'].sum()}")
print(f"Total highly suspicious (< 0.1%): {prob_analysis['Is_Highly_Suspicious'].sum()}")
print(f"\nResults based on OBEC-specific probabilities for {prob_analysis['Used_OBEC_Probability'].sum():,} cases")

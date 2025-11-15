import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import requests
    import zipfile
    import io
    from pathlib import Path
    return Path, io, mo, pd, requests, zipfile


@app.cell
def __(mo):
    mo.md(
        """
        # Top 30 Most Suspicious Zero-Vote Cases

        ## OBEC-Specific Probability Analysis

        This analysis identifies the most suspicious zero-vote cases using municipality-specific
        probabilities rather than national averages. Cases where a party gets zero votes in a
        commission are evaluated based on:

        - The party's performance specifically in that municipality (OBEC)
        - The commission size
        - Municipality context and names
        """
    )
    return


@app.cell
def __(Path, io, pd, requests, zipfile):
    # Cache file paths
    parquet_file = Path("election_data.parquet")
    municipalities_file = Path("pscoco.csv")
    parties_file = Path("cvs.csv")

    # Load election data
    if parquet_file.exists():
        df = pd.read_parquet(parquet_file)
    else:
        print("Downloading election data...")
        url_elections = "https://www.volby.cz/opendata/ps2025/csv_od/pst4p.zip"
        headers_elections = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response_elections = requests.get(url_elections, headers=headers_elections, timeout=30)
        response_elections.raise_for_status()

        zip_file = zipfile.ZipFile(io.BytesIO(response_elections.content))
        csv_filename = zip_file.namelist()[0]

        with zip_file.open(csv_filename) as f_elections:
            df = pd.read_csv(f_elections)

        df.to_parquet(parquet_file)

    print(f"Loaded {len(df):,} rows")
    return df, municipalities_file, parquet_file, parties_file


@app.cell
def __(Path, municipalities_file, parties_file, pd, requests):
    # Load municipality names
    if not municipalities_file.exists():
        print("Downloading municipality data...")
        url_municipalities = "https://www.volby.cz/opendata/ps2025/csv_od/pscoco.csv"
        headers_municipalities = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response_municipalities = requests.get(url_municipalities, headers=headers_municipalities, timeout=30)
        response_municipalities.raise_for_status()
        with open(municipalities_file, 'wb') as f_municipalities:
            f_municipalities.write(response_municipalities.content)

    municipalities = pd.read_csv(municipalities_file, encoding='utf-8')
    municipality_names = dict(zip(municipalities['OBEC'], municipalities['NAZEVOBCE']))
    print(f"Loaded {len(municipality_names):,} municipalities")

    # Load party names
    if not parties_file.exists():
        print("Downloading party data...")
        url_parties = "https://www.volby.cz/opendata/ps2025/csv_od/cvs.csv"
        headers_parties = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response_parties = requests.get(url_parties, headers=headers_parties, timeout=30)
        response_parties.raise_for_status()
        with open(parties_file, 'wb') as f_parties:
            f_parties.write(response_parties.content)

    parties = pd.read_csv(parties_file, encoding='utf-8')
    party_names = dict(zip(parties['VSTRANA'], parties['ZKRATKAV8']))
    print(f"Loaded {len(party_names):,} parties")
    return municipalities, municipality_names, parties, party_names


@app.cell
def __(df, mo, party_names, pd):
    mo.md("## Overall Party Statistics")

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

    # Display top parties with names
    top_parties_display = []
    for party_code in top_parties:
        party_name = party_names.get(party_code, str(party_code))
        pct = party_summary.loc[party_code, 'Percentage']
        votes = party_summary.loc[party_code, 'Total_Votes']
        top_parties_display.append({
            'Party': party_name,
            'Code': party_code,
            'Total_Votes': f"{votes:,.0f}",
            'Percentage': f"{pct:.2f}%"
        })

    print("Analyzing top 7 parties:")
    pd.DataFrame(top_parties_display)
    return (
        party_percentages,
        party_probabilities,
        party_summary,
        party_totals,
        top_parties,
        top_parties_display,
        total_votes,
    )


@app.cell
def __(df, mo, pd):
    mo.md("## Municipality-Specific Party Performance")

    # Calculate party performance by OBEC
    party_by_obec = df.groupby(['KSTRANA', 'OBEC'])['POC_HLASU'].agg(['sum', 'count']).reset_index()
    party_by_obec.columns = ['Party', 'OBEC', 'Total_Votes', 'Num_Commissions']

    obec_totals = df.groupby('OBEC')['POC_HLASU'].sum().reset_index()
    obec_totals.columns = ['OBEC', 'OBEC_Total_Votes']

    party_by_obec = party_by_obec.merge(obec_totals, on='OBEC')
    party_by_obec['Vote_Share_In_OBEC'] = party_by_obec['Total_Votes'] / party_by_obec['OBEC_Total_Votes'] * 100
    party_by_obec['Probability_In_OBEC'] = party_by_obec['Vote_Share_In_OBEC'] / 100

    print(f"Calculated party performance in {party_by_obec['OBEC'].nunique():,} municipalities")
    return obec_totals, party_by_obec


@app.cell
def __(df, mo, pd, top_parties):
    mo.md("## Finding Zero-Vote Cases")

    # Find zero-vote cases for top 7 parties
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

    print(f"Found {len(zero_votes_df):,} zero-vote cases for top 7 parties")
    return (
        all_combinations,
        all_commissions,
        combined,
        commission_info,
        zero_votes,
        zero_votes_df,
    )


@app.cell
def __(mo, party_by_obec, party_summary, zero_votes_df):
    mo.md("## Calculate Probabilities (OBEC-Specific)")

    # Merge OBEC-specific probabilities
    zero_votes_with_prob = zero_votes_df.merge(
        party_by_obec[['Party', 'OBEC', 'Vote_Share_In_OBEC', 'Probability_In_OBEC']],
        on=['Party', 'OBEC'],
        how='left'
    )

    # Add overall party stats
    zero_votes_with_prob['Party_Percentage'] = zero_votes_with_prob['Party'].map(
        lambda x: party_summary.loc[x, 'Percentage']
    )
    zero_votes_with_prob['Party_Probability_Overall'] = zero_votes_with_prob['Party'].map(
        lambda x: party_summary.loc[x, 'Probability']
    )

    # Calculate probabilities
    zero_votes_with_prob['Probability_of_Zero_Overall'] = (
        1 - zero_votes_with_prob['Party_Probability_Overall']
    ) ** zero_votes_with_prob['Total_Votes_In_Commission']

    zero_votes_with_prob['Probability_of_Zero_OBEC'] = (
        1 - zero_votes_with_prob['Probability_In_OBEC']
    ) ** zero_votes_with_prob['Total_Votes_In_Commission']

    zero_votes_with_prob['Used_OBEC_Probability'] = zero_votes_with_prob['Probability_In_OBEC'].notna()

    zero_votes_with_prob['Probability_of_Zero'] = zero_votes_with_prob['Probability_of_Zero_OBEC'].fillna(
        zero_votes_with_prob['Probability_of_Zero_Overall']
    )

    zero_votes_with_prob['Is_Suspicious'] = (
        (zero_votes_with_prob['Probability_of_Zero'] < 0.01) &
        zero_votes_with_prob['Used_OBEC_Probability']
    )

    zero_votes_with_prob['Is_Highly_Suspicious'] = (
        (zero_votes_with_prob['Probability_of_Zero'] < 0.001) &
        zero_votes_with_prob['Used_OBEC_Probability']
    )

    print(f"Cases with OBEC-specific data: {zero_votes_with_prob['Used_OBEC_Probability'].sum():,}")
    print(f"Suspicious cases (P < 1%): {zero_votes_with_prob['Is_Suspicious'].sum()}")
    print(f"Highly suspicious (P < 0.1%): {zero_votes_with_prob['Is_Highly_Suspicious'].sum()}")
    return (zero_votes_with_prob,)


@app.cell
def __(mo, municipality_names, party_names, pd, zero_votes_with_prob):
    mo.md("## Top 30 Most Suspicious Cases")

    # Get the most suspicious cases
    most_suspicious = zero_votes_with_prob[
        zero_votes_with_prob['Used_OBEC_Probability']
    ].sort_values('Probability_of_Zero').head(30)

    # Create enriched summary table with party and municipality names
    summary_data = []
    for idx, row in most_suspicious.iterrows():
        party_code = row['Party']
        party_name = party_names.get(party_code, str(party_code))
        obec_code = row['OBEC']
        obec_name = municipality_names.get(obec_code, f"Unknown ({obec_code})")

        status = 'HIGHLY SUSPICIOUS' if row['Is_Highly_Suspicious'] else (
            'SUSPICIOUS' if row['Is_Suspicious'] else 'Notable'
        )

        summary_data.append({
            'Rank': len(summary_data) + 1,
            'Party': party_name,
            'Party_Code': party_code,
            'Municipality': obec_name,
            'OBEC_Code': obec_code,
            'Commission_ID': row['Commission_ID'],
            'Commission_Size': int(row['Total_Votes_In_Commission']),
            'Party_Share_in_Municipality_%': f"{row['Vote_Share_In_OBEC']:.2f}",
            'Party_National_%': f"{row['Party_Percentage']:.1f}",
            'Probability_of_Zero': f"{row['Probability_of_Zero_OBEC']:.2e}",
            'Status': status
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df
    return most_suspicious, summary_data, summary_df


@app.cell
def __(mo, summary_df):
    mo.md("## Detailed View of Top 10 Cases")

    for i, row in summary_df.head(10).iterrows():
        print(f"\n{row['Rank']}. {row['Party']} (Code: {row['Party_Code']})")
        print(f"   Municipality: {row['Municipality']} (OBEC: {row['OBEC_Code']})")
        print(f"   Commission ID: {row['Commission_ID']}")
        print(f"   Commission size: {row['Commission_Size']} votes")
        print(f"   Party's share in this municipality: {row['Party_Share_in_Municipality_%']}%")
        print(f"   Party's national share: {row['Party_National_%']}%")
        print(f"   Probability of 0 votes: {row['Probability_of_Zero']}")
        print(f"   Status: {row['Status']}")
        print("-" * 100)
    return i, row


@app.cell
def __(mo, summary_df):
    mo.md("## Export Results")

    # Save to CSV for further analysis
    output_file = 'suspicious_cases_top30.csv'
    summary_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Results saved to {output_file}")
    return (output_file,)


@app.cell
def __(mo):
    mo.md(
        """
        ## Summary Statistics by Status

        Breaking down the most suspicious cases
        """
    )
    return


@app.cell
def __(summary_df):
    status_summary = summary_df.groupby('Status').agg({
        'Rank': 'count',
        'Commission_Size': ['min', 'max', 'mean']
    })
    status_summary.columns = ['Count', 'Min_Commission_Size', 'Max_Commission_Size', 'Avg_Commission_Size']
    status_summary
    return (status_summary,)


@app.cell
def __(mo):
    mo.md(
        """
        ## Key Insights

        ### What Makes a Case Suspicious?

        1. **OBEC-Specific Probability**: We use the party's actual performance in that specific
           municipality rather than national averages. This is more accurate.

        2. **Commission Size**: Larger commissions make zero votes less likely, increasing suspiciousness.

        3. **Threshold Levels**:
           - **HIGHLY SUSPICIOUS**: P < 0.1% (1 in 1,000 chance)
           - **SUSPICIOUS**: P < 1% (1 in 100 chance)

        ### What to Look For:

        - Cases where a party that performs well in a municipality gets zero votes
        - Large commissions with suspicious zeros
        - Patterns across multiple commissions in the same area
        """
    )
    return


if __name__ == "__main__":
    app.run()

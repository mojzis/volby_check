"""
Advanced Anomaly Detection for Election Data

This Marimo notebook implements multiple advanced anomaly detection techniques
to validate suspicious zero-vote cases beyond simple probability analysis.

Techniques implemented:
1. Isolation Forest (ML-based anomaly detection)
2. Multi-party zero detection
3. Commission size outlier detection
4. Party performance consistency analysis
5. Statistical outlier detection (z-scores)
6. Composite suspiciousness scoring
"""

import marimo

__generated_with = "0.9.14"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy import stats
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    import election_data_loader as edl

    # Configure pandas display for HTML export (disable pager, limit rows)
    pd.set_option('display.max_rows', 30)
    pd.set_option('display.show_dimensions', True)

    mo.md("""
    # Advanced Anomaly Detection for Czech Election Data

    This notebook applies multiple advanced statistical and machine learning techniques
    to validate suspicious zero-vote cases and identify genuine anomalies.
    """)
    return mo, pd, np, px, go, make_subplots, stats, IsolationForest, StandardScaler, edl


@app.cell
def __(mo):
    mo.md("""
    ## 1. Data Loading

    Loading election data from Czech Statistical Office (CZSO).
    """)
    return


@app.cell
def __(edl):
    # Load election data using shared function
    df = edl.load_election_data()
    print(f"Columns: {list(df.columns)}")
    return (df,)


@app.cell
def __(edl):
    # Load municipality and party data using shared functions
    obec_df = edl.load_municipality_data()
    party_df = edl.load_party_data()
    return obec_df, party_df


@app.cell
def __(mo):
    mo.md("""
    ## 2. Data Preparation

    Preparing commission-level data with all necessary features for anomaly detection.
    """)
    return


@app.cell
def __(df, pd, np):
    # Aggregate to commission level
    commission_data = df.groupby('ID_OKRSKY').agg({
        'POC_HLASU': 'sum',
        'OBEC': 'first'
    }).reset_index()
    commission_data.columns = ['ID_OKRSKY', 'TotalVotes', 'OBEC']

    # Ensure OBEC is string type to match obec_df
    commission_data['OBEC'] = commission_data['OBEC'].astype(str)

    # Calculate party-level statistics
    party_votes = df.groupby('KSTRANA')['POC_HLASU'].sum()
    total_votes = df['POC_HLASU'].sum()
    party_probs = party_votes / total_votes

    # Identify major parties (>5% nationally)
    major_parties = party_probs[party_probs > 0.05].index.tolist()

    print(f"Total commissions: {len(commission_data):,}")
    print(f"Total votes cast: {total_votes:,}")
    print(f"Major parties (>5%): {len(major_parties)}")
    print(f"Major party codes: {major_parties}")

    # Create wide format: each row is a commission, each column is a party's votes
    commission_party_votes = df.pivot_table(
        index='ID_OKRSKY',
        columns='KSTRANA',
        values='POC_HLASU',
        fill_value=0
    ).reset_index()

    # Add OBEC information
    commission_party_votes_with_obec = commission_party_votes.merge(
        commission_data[['ID_OKRSKY', 'OBEC', 'TotalVotes']],
        on='ID_OKRSKY'
    )

    print(f"Commission-party matrix shape: {commission_party_votes_with_obec.shape}")
    return commission_data, party_votes, total_votes, party_probs, major_parties, commission_party_votes_with_obec


@app.cell
def __(mo):
    mo.md("""
    ## 2.1 Commission Size Filtering

    Focusing on statistically meaningful commission sizes (≥150 votes).

    **Why 150 votes?**
    - 25th percentile of commission sizes (excludes smallest 25%)
    - Large enough for major parties (5-35% support) to have statistically significant absence
    - For a party with 10% support, P(0 votes) in a 150-vote commission ≈ 0.000003%
    - Avoids false positives from natural variance in very small commissions

    **Impact:**
    - Very small commissions (< 150 votes) have high natural variance
    - ML anomaly detection can mistake normal variance for suspicious patterns
    - By filtering, we focus on cases where anomalies are meaningful
    """)
    return


@app.cell
def __(commission_party_votes_with_obec, pd, np):
    # Filter to commissions with >= 150 votes
    MIN_COMMISSION_SIZE = 150

    commission_filtered = commission_party_votes_with_obec[
        commission_party_votes_with_obec['TotalVotes'] >= MIN_COMMISSION_SIZE
    ].copy()

    # Calculate filtering statistics
    total_commissions_before = len(commission_party_votes_with_obec)
    total_commissions_after = len(commission_filtered)
    filtered_out = total_commissions_before - total_commissions_after
    filtered_pct = (filtered_out / total_commissions_before) * 100

    print(f"Commission Size Filter: >= {MIN_COMMISSION_SIZE} votes")
    print(f"Before filtering: {total_commissions_before:,} commissions")
    print(f"After filtering: {total_commissions_after:,} commissions")
    print(f"Filtered out: {filtered_out:,} commissions ({filtered_pct:.1f}%)")
    print(f"\nFiltered commission stats:")
    print(f"  Min votes: {commission_filtered['TotalVotes'].min():.0f}")
    print(f"  Median votes: {commission_filtered['TotalVotes'].median():.0f}")
    print(f"  Mean votes: {commission_filtered['TotalVotes'].mean():.0f}")
    print(f"  Max votes: {commission_filtered['TotalVotes'].max():.0f}")

    return MIN_COMMISSION_SIZE, commission_filtered, total_commissions_before, total_commissions_after, filtered_out, filtered_pct


@app.cell
def __(mo):
    mo.md("""
    ## 3. Advanced Anomaly Detection Techniques

    ### 3.1 Multi-Party Zero Detection

    Detecting commissions where multiple major parties received 0 votes.
    This is highly unusual and suggests potential data quality issues.
    """)
    return


@app.cell
def __(commission_filtered, major_parties, pd):
    # Count zeros for major parties in each commission
    major_party_cols = [col for col in major_parties if col in commission_filtered.columns]

    commission_with_zeros = commission_filtered.copy()
    commission_with_zeros['ZeroCount_MajorParties'] = (
        commission_with_zeros[major_party_cols] == 0
    ).sum(axis=1)

    # Flag suspicious multi-party zeros
    commission_with_zeros['MultiPartyZero_Flag'] = (
        commission_with_zeros['ZeroCount_MajorParties'] >= 2
    )

    multi_zero_summary = pd.DataFrame({
        'Zero_Count': range(0, commission_with_zeros['ZeroCount_MajorParties'].max() + 1),
        'Commissions': [
            (commission_with_zeros['ZeroCount_MajorParties'] == i).sum()
            for i in range(0, commission_with_zeros['ZeroCount_MajorParties'].max() + 1)
        ]
    })

    print("Multi-Party Zero Distribution:")
    print(multi_zero_summary)
    print(f"\nCommissions with 2+ major parties at zero: {commission_with_zeros['MultiPartyZero_Flag'].sum():,}")
    return major_party_cols, multi_zero_summary, commission_with_zeros


@app.cell
def __(px, multi_zero_summary):
    # Visualize multi-party zero distribution
    fig_multi_zero = px.bar(
        multi_zero_summary,
        x='Zero_Count',
        y='Commissions',
        title='Distribution of Major Party Zeros per Commission',
        labels={'Zero_Count': 'Number of Major Parties with 0 Votes', 'Commissions': 'Number of Commissions'},
        text='Commissions'
    )
    fig_multi_zero.update_traces(textposition='outside')
    fig_multi_zero
    return (fig_multi_zero,)


@app.cell
def __(mo):
    mo.md("""
    ### 3.2 Commission Size Outlier Detection

    Identifying commissions with abnormally large or small voter counts
    compared to others in the same municipality.
    """)
    return


@app.cell
def __(commission_with_zeros, np):
    # Calculate municipality-level statistics
    obec_stats = commission_with_zeros.groupby('OBEC').agg({
        'TotalVotes': ['mean', 'std', 'count']
    }).reset_index()
    obec_stats.columns = ['OBEC', 'OBEC_MeanVotes', 'OBEC_StdVotes', 'OBEC_CommissionCount']

    # Merge back to commissions
    commission_with_stats = commission_with_zeros.merge(obec_stats, on='OBEC')

    # Calculate z-score for commission size within municipality
    commission_with_stats['CommissionSize_ZScore'] = np.where(
        commission_with_stats['OBEC_StdVotes'] > 0,
        (commission_with_stats['TotalVotes'] - commission_with_stats['OBEC_MeanVotes']) / commission_with_stats['OBEC_StdVotes'],
        0
    )

    # Flag extreme outliers (|z| > 3)
    commission_with_stats['CommissionSize_Outlier'] = (
        np.abs(commission_with_stats['CommissionSize_ZScore']) > 3
    )

    print(f"Commissions with abnormal size (|z| > 3): {commission_with_stats['CommissionSize_Outlier'].sum():,}")
    print(f"Mean z-score: {commission_with_stats['CommissionSize_ZScore'].mean():.2f}")
    print(f"Std z-score: {commission_with_stats['CommissionSize_ZScore'].std():.2f}")
    return obec_stats, commission_with_stats


@app.cell
def __(px, commission_with_stats):
    # Visualize commission size distribution
    fig_size = px.histogram(
        commission_with_stats,
        x='CommissionSize_ZScore',
        nbins=100,
        title='Distribution of Commission Size Z-Scores (within municipality)',
        labels={'CommissionSize_ZScore': 'Z-Score', 'count': 'Number of Commissions'}
    )
    fig_size.add_vline(x=-3, line_dash="dash", line_color="red", annotation_text="Outlier threshold")
    fig_size.add_vline(x=3, line_dash="dash", line_color="red")
    fig_size
    return (fig_size,)


@app.cell
def __(mo):
    mo.md("""
    ### 3.3 Party Performance Consistency Analysis

    Analyzing variance in party performance across commissions within the same municipality.
    High variance may indicate irregularities.
    """)
    return


@app.cell
def __(commission_with_zeros, major_party_cols, pd, np):
    # Calculate coefficient of variation for each party in each municipality
    consistency_data = []

    for obec in commission_with_zeros['OBEC'].unique():
        obec_commissions = commission_with_zeros[commission_with_zeros['OBEC'] == obec]

        if len(obec_commissions) < 2:  # Need at least 2 commissions to calculate variance
            continue

        for _party in major_party_cols:
            votes = obec_commissions[_party]
            _total_votes = obec_commissions['TotalVotes']

            # Calculate vote share for each commission
            vote_shares = votes / _total_votes.replace(0, np.nan)
            vote_shares = vote_shares.dropna()

            if len(vote_shares) < 2:
                continue

            mean_share = vote_shares.mean()
            std_share = vote_shares.std()
            cv = std_share / mean_share if mean_share > 0 else np.nan

            consistency_data.append({
                'OBEC': obec,
                'Party': _party,
                'MeanShare': mean_share,
                'StdShare': std_share,
                'CV': cv,
                'CommissionCount': len(vote_shares),
                'ZeroCount': (votes == 0).sum()
            })

    consistency_df = pd.DataFrame(consistency_data)

    # Flag high variance cases (CV > 0.5 and party has reasonable support)
    consistency_df['HighVariance_Flag'] = (
        (consistency_df['CV'] > 0.5) &
        (consistency_df['MeanShare'] > 0.05) &
        (consistency_df['CommissionCount'] >= 3)
    )

    print(f"Party-municipality combinations analyzed: {len(consistency_df):,}")
    print(f"High variance cases: {consistency_df['HighVariance_Flag'].sum():,}")
    return consistency_data, consistency_df


@app.cell
def __(px, consistency_df):
    # Visualize CV distribution
    fig_cv = px.histogram(
        consistency_df[consistency_df['CV'].notna() & (consistency_df['CV'] < 2)],
        x='CV',
        nbins=100,
        title='Distribution of Coefficient of Variation for Party Performance',
        labels={'CV': 'Coefficient of Variation', 'count': 'Number of Cases'}
    )
    fig_cv.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="High variance threshold")
    fig_cv
    return (fig_cv,)


@app.cell
def __(mo):
    mo.md("""
    ### 3.4 Isolation Forest Anomaly Detection

    Using machine learning (Isolation Forest) to detect anomalies based on
    multiple features simultaneously.
    """)
    return


@app.cell
def __(commission_with_stats, major_party_cols, IsolationForest, StandardScaler, np, pd):
    # Prepare features for Isolation Forest
    # Enhanced features to detect patterns similar to probability-based suspiciousness

    X_features = commission_with_stats.copy()

    # 1. Vote shares for major parties
    for _p in major_party_cols:
        X_features[f'{_p}_share'] = X_features[_p] / X_features['TotalVotes'].replace(0, np.nan)

    share_cols = [f'{_p}_share' for _p in major_party_cols]

    # 2. Deviation from municipality mean for each party
    # This helps detect commissions that are outliers within their municipality
    for _p in major_party_cols:
        obec_party_mean = X_features.groupby('OBEC')[f'{_p}_share'].transform('mean')
        X_features[f'{_p}_deviation'] = X_features[f'{_p}_share'] - obec_party_mean

    deviation_cols = [f'{_p}_deviation' for _p in major_party_cols]

    # 3. Number of major parties with unusually low support (< municipality average)
    X_features['UnusuallyLowParties'] = 0
    for _p in major_party_cols:
        obec_party_mean = X_features.groupby('OBEC')[f'{_p}_share'].transform('mean')
        obec_party_std = X_features.groupby('OBEC')[f'{_p}_share'].transform('std')
        # Count parties that are > 2 std below municipality mean
        is_unusually_low = (X_features[f'{_p}_share'] < obec_party_mean - 2 * obec_party_std)
        X_features['UnusuallyLowParties'] += is_unusually_low.astype(int)

    # Select features for modeling
    model_features = (
        share_cols +
        deviation_cols +
        ['TotalVotes', 'ZeroCount_MajorParties', 'CommissionSize_ZScore', 'UnusuallyLowParties']
    )

    X = X_features[model_features].fillna(0)

    print(f"Enhanced feature engineering:")
    print(f"  Vote shares: {len(share_cols)} features")
    print(f"  Municipality deviations: {len(deviation_cols)} features")
    print(f"  Other features: 4")
    print(f"  Total features: {len(model_features)}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Isolation Forest
    iso_forest = IsolationForest(
        contamination=0.05,  # Expect 5% anomalies
        random_state=42,
        n_estimators=100
    )

    # Predict anomalies
    X_features['IsolationForest_Prediction'] = iso_forest.fit_predict(X_scaled)
    X_features['IsolationForest_Score'] = iso_forest.score_samples(X_scaled)

    # -1 = anomaly, 1 = normal
    X_features['IsolationForest_Anomaly'] = (
        X_features['IsolationForest_Prediction'] == -1
    )

    print(f"Anomalies detected by Isolation Forest: {X_features['IsolationForest_Anomaly'].sum():,}")
    print(f"Anomaly rate: {X_features['IsolationForest_Anomaly'].mean():.2%}")
    return X_features, share_cols, deviation_cols, model_features, X, scaler, X_scaled, iso_forest


@app.cell
def __(px, X_features):
    # Visualize Isolation Forest scores
    fig_iso = px.histogram(
        X_features,
        x='IsolationForest_Score',
        nbins=100,
        title='Distribution of Isolation Forest Anomaly Scores',
        labels={'IsolationForest_Score': 'Anomaly Score (lower = more anomalous)', 'count': 'Number of Commissions'},
        color='IsolationForest_Anomaly',
        color_discrete_map={True: 'red', False: 'blue'}
    )
    fig_iso
    return (fig_iso,)


@app.cell
def __(mo):
    mo.md("""
    ## 4. Composite Suspiciousness Score

    Combining all anomaly indicators into a single composite score.
    """)
    return


@app.cell
def __(X_features, np):
    # Calculate composite score
    # Normalize each indicator to 0-1 scale and weight them

    weights = {
        'multi_party_zero': 3.0,      # High weight - very suspicious
        'commission_size': 1.0,         # Medium weight
        'isolation_forest': 2.0         # High weight - ML-based
    }

    commission_final = X_features.copy()

    # Multi-party zero score (0-1, higher = more zeros)
    max_zeros = commission_final['ZeroCount_MajorParties'].max()
    commission_final['Score_MultiPartyZero'] = (
        commission_final['ZeroCount_MajorParties'] / max_zeros
    )

    # Commission size score (0-1, based on absolute z-score)
    commission_final['Score_CommissionSize'] = np.minimum(
        np.abs(commission_final['CommissionSize_ZScore']) / 5,  # Cap at z=5
        1.0
    )

    # Isolation Forest score (0-1, inverted and normalized)
    # More negative scores = more anomalous
    iso_min = commission_final['IsolationForest_Score'].min()
    iso_max = commission_final['IsolationForest_Score'].max()
    commission_final['Score_IsolationForest'] = (
        (iso_max - commission_final['IsolationForest_Score']) / (iso_max - iso_min)
    )

    # Calculate weighted composite score
    commission_final['CompositeScore'] = (
        weights['multi_party_zero'] * commission_final['Score_MultiPartyZero'] +
        weights['commission_size'] * commission_final['Score_CommissionSize'] +
        weights['isolation_forest'] * commission_final['Score_IsolationForest']
    ) / sum(weights.values())

    # Rank by composite score
    commission_final['SuspiciousnessRank'] = (
        commission_final['CompositeScore'].rank(ascending=False, method='min').astype(int)
    )

    print(f"Composite score range: {commission_final['CompositeScore'].min():.4f} to {commission_final['CompositeScore'].max():.4f}")
    print(f"Mean composite score: {commission_final['CompositeScore'].mean():.4f}")
    print(f"Median composite score: {commission_final['CompositeScore'].median():.4f}")
    return weights, max_zeros, iso_min, iso_max, commission_final


@app.cell
def __(px, commission_final):
    # Visualize composite score distribution
    fig_composite = px.histogram(
        commission_final,
        x='CompositeScore',
        nbins=100,
        title='Distribution of Composite Suspiciousness Scores',
        labels={'CompositeScore': 'Composite Score (0-1)', 'count': 'Number of Commissions'}
    )

    # Add percentile lines
    p95 = commission_final['CompositeScore'].quantile(0.95)
    p99 = commission_final['CompositeScore'].quantile(0.99)

    fig_composite.add_vline(x=p95, line_dash="dash", line_color="orange", annotation_text="95th percentile")
    fig_composite.add_vline(x=p99, line_dash="dash", line_color="red", annotation_text="99th percentile")
    fig_composite
    return fig_composite, p95, p99


@app.cell
def __(mo):
    mo.md("""
    ## 5. Top Suspicious Commissions

    Identifying the most suspicious commissions based on composite scoring.
    """)
    return


@app.cell
def __(commission_final, obec_df, pd):
    # Get top 100 most suspicious commissions
    top_suspicious = commission_final.nlargest(100, 'CompositeScore').copy()

    # Add municipality names if available
    obec_merge_cols = []
    if 'OBEC' in obec_df.columns:
        obec_merge_cols.append('OBEC')
    # Check for municipality name column (could be NAZEVOBCE, NAZEV, or similar)
    obec_name_col_found = None
    for possible_obec_col in ['NAZEVOBCE', 'NAZEV', 'NAZEVOB', 'OBEC_NAZEV']:
        if possible_obec_col in obec_df.columns:
            obec_name_col_found = possible_obec_col
            break

    if obec_merge_cols and obec_name_col_found:
        obec_merge_cols.append(obec_name_col_found)
        top_suspicious = top_suspicious.merge(
            obec_df[obec_merge_cols],
            on='OBEC',
            how='left'
        )
        # Rename to standard name for consistency
        if obec_name_col_found != 'NAZEVOBCE':
            top_suspicious = top_suspicious.rename(columns={obec_name_col_found: 'NAZEVOBCE'})
    else:
        # If no name column found, use OBEC as the display name
        top_suspicious['NAZEVOBCE'] = top_suspicious['OBEC']

    # Select relevant columns for display
    display_cols = [
        'SuspiciousnessRank',
        'ID_OKRSKY',
        'NAZEVOBCE',
        'CompositeScore',
        'TotalVotes',
        'ZeroCount_MajorParties',
        'CommissionSize_ZScore',
        'IsolationForest_Anomaly',
        'MultiPartyZero_Flag',
        'CommissionSize_Outlier'
    ]

    top_suspicious_display = top_suspicious[display_cols].copy()

    # Rename for readability
    top_suspicious_display.columns = [
        'Rank', 'Commission ID', 'Municipality', 'Composite Score',
        'Total Votes', 'Major Parties w/ 0', 'Size Z-Score',
        'ML Anomaly', 'Multi-Zero', 'Size Outlier'
    ]

    print(f"Top 100 most suspicious commissions:")
    top_suspicious_display.head(20)
    return top_suspicious, top_suspicious_display, display_cols


@app.cell
def __(mo):
    mo.md("""
    ## 6. Anomaly Type Analysis

    Breaking down what types of anomalies are most common in highly suspicious cases.
    """)
    return


@app.cell
def __(commission_final, pd):
    # Analyze top 5% most suspicious
    top_5pct_threshold = commission_final['CompositeScore'].quantile(0.95)
    top_5pct = commission_final[commission_final['CompositeScore'] >= top_5pct_threshold]

    anomaly_breakdown = pd.DataFrame({
        'Anomaly Type': [
            'Multi-Party Zero (2+)',
            'Commission Size Outlier',
            'Isolation Forest Anomaly',
            'High Variance (from consistency analysis)'
        ],
        'Count in Top 5%': [
            top_5pct['MultiPartyZero_Flag'].sum(),
            top_5pct['CommissionSize_Outlier'].sum(),
            top_5pct['IsolationForest_Anomaly'].sum(),
            0  # Will update this if we add the flag
        ],
        'Percentage': [
            top_5pct['MultiPartyZero_Flag'].mean() * 100,
            top_5pct['CommissionSize_Outlier'].mean() * 100,
            top_5pct['IsolationForest_Anomaly'].mean() * 100,
            0
        ]
    })

    print(f"Total commissions in top 5%: {len(top_5pct):,}")
    print(f"Threshold score: {top_5pct_threshold:.4f}")
    print("\nAnomaly type breakdown:")
    anomaly_breakdown
    return anomaly_breakdown, top_5pct_threshold, top_5pct


@app.cell
def __(px, anomaly_breakdown):
    # Visualize anomaly types
    fig_breakdown = px.bar(
        anomaly_breakdown[anomaly_breakdown['Count in Top 5%'] > 0],
        x='Anomaly Type',
        y='Count in Top 5%',
        title='Anomaly Types in Top 5% Most Suspicious Commissions',
        text='Count in Top 5%'
    )
    fig_breakdown.update_traces(textposition='outside')
    fig_breakdown
    return (fig_breakdown,)


@app.cell
def __(mo):
    mo.md("""
    ## 7. Party-Specific Zero Analysis with Validation

    Examining zero-vote cases for major parties and validating with our anomaly scores.
    """)
    return


@app.cell
def __(df, major_party_cols, commission_final, party_df, obec_df, pd):
    # Find all zero-vote cases for major parties
    zero_cases = []

    for _pt in major_party_cols:
        # Find commissions where this party got 0 votes
        party_data = df[df['KSTRANA'] == _pt][['ID_OKRSKY', 'POC_HLASU', 'OBEC']]

        # Get all commissions
        all_commissions = set(commission_final['ID_OKRSKY'])
        party_commissions = set(party_data['ID_OKRSKY'])

        # Commissions where party got 0 (either not in data or POC_HLASU = 0)
        zero_commissions = all_commissions - party_commissions
        zero_commissions.update(party_data[party_data['POC_HLASU'] == 0]['ID_OKRSKY'])

        for commission in zero_commissions:
            comm_data = commission_final[commission_final['ID_OKRSKY'] == commission]
            if len(comm_data) == 0:
                continue

            comm_data = comm_data.iloc[0]

            zero_cases.append({
                'Party': _pt,
                'ID_OKRSKY': commission,
                'OBEC': comm_data['OBEC'],
                'TotalVotes': comm_data['TotalVotes'],
                'CompositeScore': comm_data['CompositeScore'],
                'SuspiciousnessRank': comm_data['SuspiciousnessRank'],
                'ZeroCount_MajorParties': comm_data['ZeroCount_MajorParties'],
                'IsolationForest_Anomaly': comm_data['IsolationForest_Anomaly'],
                'CommissionSize_Outlier': comm_data['CommissionSize_Outlier']
            })

    zero_cases_df = pd.DataFrame(zero_cases)

    # Ensure Party column is string type to match party_df['KSTRANA']
    zero_cases_df['Party'] = zero_cases_df['Party'].astype(str)

    # Add party names if available
    # Check for party short name column (could be ZKRATKAK8, ZKRATKA, etc.)
    party_name_col_found = None
    for possible_party_col in ['ZKRATKAK8', 'ZKRATKA', 'ZKRATKAK30', 'NAZEVSTR']:
        if possible_party_col in party_df.columns:
            party_name_col_found = possible_party_col
            break

    if 'KSTRANA' in party_df.columns and party_name_col_found:
        party_merge_df = party_df[['KSTRANA', party_name_col_found]].copy()
        zero_cases_df = zero_cases_df.merge(
            party_merge_df,
            left_on='Party',
            right_on='KSTRANA',
            how='left'
        )
        # Rename to standard name for consistency
        if party_name_col_found != 'ZKRATKAK8':
            zero_cases_df = zero_cases_df.rename(columns={party_name_col_found: 'ZKRATKAK8'})
    else:
        # If no party name found, use party code as the display name
        zero_cases_df['ZKRATKAK8'] = zero_cases_df['Party']

    # Add municipality names if available
    obec_name_col_for_zeros = None
    for possible_obec_col_zero in ['NAZEVOBCE', 'NAZEV', 'NAZEVOB', 'OBEC_NAZEV']:
        if possible_obec_col_zero in obec_df.columns:
            obec_name_col_for_zeros = possible_obec_col_zero
            break

    if 'OBEC' in obec_df.columns and obec_name_col_for_zeros:
        obec_merge_df = obec_df[['OBEC', obec_name_col_for_zeros]].copy()
        zero_cases_df = zero_cases_df.merge(
            obec_merge_df,
            on='OBEC',
            how='left'
        )
        # Rename to standard name for consistency
        if obec_name_col_for_zeros != 'NAZEVOBCE':
            zero_cases_df = zero_cases_df.rename(columns={obec_name_col_for_zeros: 'NAZEVOBCE'})
    else:
        # If no name column found, use OBEC as the display name
        zero_cases_df['NAZEVOBCE'] = zero_cases_df['OBEC']

    print(f"Total zero-vote cases for major parties: {len(zero_cases_df):,}")
    print(f"\nZero cases by party:")
    if 'ZKRATKAK8' in zero_cases_df.columns:
        print(zero_cases_df.groupby('ZKRATKAK8').size().sort_values(ascending=False))
    else:
        print(zero_cases_df.groupby('Party').size().sort_values(ascending=False))
    return zero_cases, zero_cases_df


@app.cell
def __(zero_cases_df):
    # Analyze validation metrics for zero-vote cases
    print("\nValidation of Zero-Vote Cases:")
    print(f"Zeros with high composite score (>0.5): {(zero_cases_df['CompositeScore'] > 0.5).sum():,} ({(zero_cases_df['CompositeScore'] > 0.5).mean():.1%})")
    print(f"Zeros flagged as ML anomaly: {zero_cases_df['IsolationForest_Anomaly'].sum():,} ({zero_cases_df['IsolationForest_Anomaly'].mean():.1%})")
    print(f"Zeros with multi-party zeros (2+): {(zero_cases_df['ZeroCount_MajorParties'] >= 2).sum():,} ({(zero_cases_df['ZeroCount_MajorParties'] >= 2).mean():.1%})")
    print(f"Zeros in size-outlier commissions: {zero_cases_df['CommissionSize_Outlier'].sum():,} ({zero_cases_df['CommissionSize_Outlier'].mean():.1%})")

    print(f"\nMean composite score for zero cases: {zero_cases_df['CompositeScore'].mean():.4f}")
    print(f"Median composite score for zero cases: {zero_cases_df['CompositeScore'].median():.4f}")
    return


@app.cell
def __(px, zero_cases_df):
    # Visualize composite scores for zero-vote cases
    fig_zero_scores = px.box(
        zero_cases_df,
        x='ZKRATKAK8',
        y='CompositeScore',
        title='Composite Suspiciousness Scores for Zero-Vote Cases by Party',
        labels={'ZKRATKAK8': 'Party', 'CompositeScore': 'Composite Score'}
    )
    fig_zero_scores.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="High suspicion threshold")
    fig_zero_scores
    return (fig_zero_scores,)


@app.cell
def __(mo):
    mo.md("""
    ## 8. Most Validated Suspicious Cases

    Showing zero-vote cases with the highest composite anomaly scores.
    These are the cases that multiple independent validation methods agree are suspicious.
    """)
    return


@app.cell
def __(zero_cases_df):
    # Get top suspicious zero cases
    most_suspicious_zeros = zero_cases_df.nlargest(50, 'CompositeScore')

    # Select columns for display
    display_zero_cols = [
        'SuspiciousnessRank',
        'ZKRATKAK8',
        'NAZEVOBCE',
        'ID_OKRSKY',
        'TotalVotes',
        'CompositeScore',
        'ZeroCount_MajorParties',
        'IsolationForest_Anomaly',
        'CommissionSize_Outlier'
    ]

    most_suspicious_zeros_display = most_suspicious_zeros[display_zero_cols].copy()
    most_suspicious_zeros_display.columns = [
        'Overall Rank', 'Party', 'Municipality', 'Commission ID',
        'Total Votes', 'Composite Score', 'Major Parties w/ 0',
        'ML Anomaly', 'Size Outlier'
    ]

    print("Top 50 Most Validated Suspicious Zero-Vote Cases:")
    most_suspicious_zeros_display.head(25)
    return most_suspicious_zeros, display_zero_cols, most_suspicious_zeros_display


@app.cell
def __(mo):
    mo.md("""
    ## 9. Summary Statistics

    Overall validation summary comparing probability-based suspicions with multi-method validation.
    """)
    return


@app.cell
def __(commission_final, zero_cases_df, pd):
    # Summary statistics
    summary_stats = pd.DataFrame({
        'Metric': [
            'Total Commissions',
            'Commissions with Any Major Party Zero',
            'Commissions with 2+ Major Party Zeros',
            'Commissions Flagged by Isolation Forest',
            'Commissions with Abnormal Size',
            'Commissions in Top 5% Composite Score',
            'Zero-Vote Cases for Major Parties',
            'Zero Cases with High Composite Score (>0.5)',
            'Zero Cases Validated by ML (Isolation Forest)',
        ],
        'Count': [
            len(commission_final),
            (commission_final['ZeroCount_MajorParties'] > 0).sum(),
            commission_final['MultiPartyZero_Flag'].sum(),
            commission_final['IsolationForest_Anomaly'].sum(),
            commission_final['CommissionSize_Outlier'].sum(),
            (commission_final['CompositeScore'] >= commission_final['CompositeScore'].quantile(0.95)).sum(),
            len(zero_cases_df),
            (zero_cases_df['CompositeScore'] > 0.5).sum(),
            zero_cases_df['IsolationForest_Anomaly'].sum(),
        ]
    })

    summary_stats['Percentage'] = (summary_stats['Count'] / len(commission_final) * 100).round(2)

    print("Summary of Advanced Anomaly Detection Results:")
    summary_stats
    return (summary_stats,)


@app.cell
def __(mo):
    mo.md("""
    ## 9.1 Method Comparison: Anomaly Detection vs Probability Analysis

    Comparing how well the different anomaly detection methods agree with each other.
    This helps identify cases where **multiple independent methods** converge on the same
    suspicious commissions - providing stronger evidence than any single method alone.

    **Key Questions:**
    - Do the ML-based anomalies overlap with multi-party zeros?
    - Are high composite scores driven by multiple factors or just one?
    - Which method is most selective vs. most comprehensive?
    """)
    return


@app.cell
def __(commission_final, pd, px):
    # Create a comparison of different detection methods
    # Define thresholds for each method
    high_composite_threshold = commission_final['CompositeScore'].quantile(0.95)  # Top 5%

    # Flag commissions by different criteria
    comparison_df = commission_final[['ID_OKRSKY', 'OBEC', 'TotalVotes', 'CompositeScore']].copy()

    comparison_df['Method_MultiPartyZero'] = commission_final['MultiPartyZero_Flag']
    comparison_df['Method_IsolationForest'] = commission_final['IsolationForest_Anomaly']
    comparison_df['Method_SizeOutlier'] = commission_final['CommissionSize_Outlier']
    comparison_df['Method_HighComposite'] = commission_final['CompositeScore'] >= high_composite_threshold

    # Count how many methods flag each commission
    method_cols = ['Method_MultiPartyZero', 'Method_IsolationForest', 'Method_SizeOutlier', 'Method_HighComposite']
    comparison_df['Methods_Agreement'] = comparison_df[method_cols].sum(axis=1)

    # Create overlap statistics
    overlap_stats = pd.DataFrame({
        'Methods Agreeing': range(0, 5),
        'Commissions': [
            (comparison_df['Methods_Agreement'] == i).sum()
            for i in range(0, 5)
        ]
    })
    overlap_stats['Percentage'] = (overlap_stats['Commissions'] / len(comparison_df) * 100).round(2)

    print("Method Agreement Distribution:")
    print(overlap_stats)
    print(f"\nCommissions flagged by 2+ methods: {(comparison_df['Methods_Agreement'] >= 2).sum():,}")
    print(f"Commissions flagged by 3+ methods: {(comparison_df['Methods_Agreement'] >= 3).sum():,}")
    print(f"Commissions flagged by all 4 methods: {(comparison_df['Methods_Agreement'] == 4).sum():,}")

    # Visualize overlap
    fig_overlap = px.bar(
        overlap_stats,
        x='Methods Agreeing',
        y='Commissions',
        title='Number of Detection Methods Agreeing per Commission',
        labels={'Methods Agreeing': 'Number of Methods Flagging Commission', 'Commissions': 'Count'},
        text='Commissions'
    )
    fig_overlap.update_traces(textposition='outside')
    fig_overlap

    return comparison_df, method_cols, overlap_stats, high_composite_threshold, fig_overlap


@app.cell
def __(pd, comparison_df, method_cols):
    # Pairwise method comparison
    # Create a confusion matrix showing overlap between methods

    pairwise_comparison = []
    for i, method1 in enumerate(method_cols):
        for method2 in method_cols[i+1:]:
            both = (comparison_df[method1] & comparison_df[method2]).sum()
            only_1 = (comparison_df[method1] & ~comparison_df[method2]).sum()
            only_2 = (~comparison_df[method1] & comparison_df[method2]).sum()
            neither = (~comparison_df[method1] & ~comparison_df[method2]).sum()

            total_method1 = comparison_df[method1].sum()
            total_method2 = comparison_df[method2].sum()

            overlap_rate = both / min(total_method1, total_method2) * 100 if min(total_method1, total_method2) > 0 else 0

            pairwise_comparison.append({
                'Method 1': method1.replace('Method_', ''),
                'Method 2': method2.replace('Method_', ''),
                'Both Flag': both,
                'Only Method 1': only_1,
                'Only Method 2': only_2,
                'Overlap %': f"{overlap_rate:.1f}%"
            })

    pairwise_df = pd.DataFrame(pairwise_comparison)

    print("\nPairwise Method Overlap:")
    print("(Shows how many commissions are flagged by both methods)")
    pairwise_df

    return pairwise_comparison, pairwise_df


@app.cell
def __(comparison_df, commission_final, obec_df, pd):
    # Show high-consensus cases (flagged by 3+ methods)
    high_consensus = comparison_df[comparison_df['Methods_Agreement'] >= 3].copy()

    # Add municipality names
    obec_name_col_consensus = None
    for possible_col in ['NAZEVOBCE', 'NAZEV', 'NAZEVOB', 'OBEC_NAZEV']:
        if possible_col in obec_df.columns:
            obec_name_col_consensus = possible_col
            break

    if 'OBEC' in obec_df.columns and obec_name_col_consensus:
        high_consensus = high_consensus.merge(
            obec_df[['OBEC', obec_name_col_consensus]],
            on='OBEC',
            how='left'
        )
        if obec_name_col_consensus != 'NAZEVOBCE':
            high_consensus = high_consensus.rename(columns={obec_name_col_consensus: 'NAZEVOBCE'})
    else:
        high_consensus['NAZEVOBCE'] = high_consensus['OBEC']

    # Add zero count info
    high_consensus = high_consensus.merge(
        commission_final[['ID_OKRSKY', 'ZeroCount_MajorParties', 'SuspiciousnessRank']],
        on='ID_OKRSKY',
        how='left'
    )

    high_consensus_display = high_consensus[[
        'SuspiciousnessRank', 'ID_OKRSKY', 'NAZEVOBCE', 'TotalVotes',
        'CompositeScore', 'Methods_Agreement', 'ZeroCount_MajorParties',
        'Method_MultiPartyZero', 'Method_IsolationForest', 'Method_SizeOutlier', 'Method_HighComposite'
    ]].copy()

    high_consensus_display = high_consensus_display.sort_values('CompositeScore', ascending=False)

    high_consensus_display.columns = [
        'Rank', 'Commission ID', 'Municipality', 'Total Votes',
        'Composite Score', '# Methods', 'Major Parties w/ 0',
        'Multi-Zero', 'ML Anomaly', 'Size Outlier', 'High Composite'
    ]

    print(f"\nHigh-Consensus Cases (3+ methods agree): {len(high_consensus):,} commissions")
    print("These are the MOST SUSPICIOUS cases with multiple independent validations:\n")
    high_consensus_display.head(30)

    return high_consensus, high_consensus_display, obec_name_col_consensus


@app.cell
def __(mo):
    mo.md("""
    ### Interpretation of Method Comparison

    **Strong Evidence (3-4 methods agree):**
    - These commissions have unusual patterns detected by multiple independent approaches
    - Very unlikely to be random statistical fluctuations
    - Worthy of detailed investigation

    **Moderate Evidence (2 methods agree):**
    - Some unusual characteristics, but not extreme
    - Could be real anomalies or edge cases
    - Consider in context with other information

    **Weak Evidence (1 method only):**
    - Flagged by only one approach
    - May be method-specific artifacts or natural variance
    - Less reliable without corroboration

    **To compare with Probability-Based Analysis:**
    Run `suspicious_zeros_with_obec_analysis.py` separately and manually compare
    commission IDs with high probability suspiciousness to these anomaly-detected cases.
    Cases that appear in **both analyses** have the strongest evidence.
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ## 10. Export Results

    Saving the validated suspicious cases for further analysis.
    """)
    return


@app.cell
def __(commission_final, zero_cases_df, most_suspicious_zeros, high_consensus):
    # Export commission-level data with all scores
    commission_final.to_csv('advanced_anomaly_commission_scores.csv', index=False)
    print("Saved: advanced_anomaly_commission_scores.csv")

    # Export all zero cases with validation
    zero_cases_df.to_csv('zero_cases_with_validation.csv', index=False)
    print("Saved: zero_cases_with_validation.csv")

    # Export top suspicious zeros
    most_suspicious_zeros.to_csv('most_suspicious_zero_cases.csv', index=False)
    print("Saved: most_suspicious_zero_cases.csv")

    # Export high-consensus cases (3+ methods agree)
    high_consensus.to_csv('high_consensus_cases.csv', index=False)
    print("Saved: high_consensus_cases.csv")

    print("\nAll validation data exported successfully!")
    return


@app.cell
def __(mo):
    mo.md("""
    ## Conclusions

    This advanced anomaly detection analysis provides **independent validation** separate from
    probability-based methods, using multiple complementary techniques:

    ### Key Improvements

    1. **Commission Size Filtering (≥150 votes)**
       - Focuses on statistically meaningful commission sizes
       - Eliminates noise from very small commissions with natural high variance
       - Aligns the analysis population with probability-based approaches

    2. **Enhanced Feature Engineering**
       - Party performance deviations from municipality averages
       - Detection of unusually low support patterns
       - Multi-dimensional anomaly signals (not just vote counts)

    3. **Multiple Independent Detection Methods**
       - **Multi-Party Zero Detection**: Simultaneous zeros for multiple major parties
       - **Commission Size Outlier Detection**: Abnormal commission sizes within municipalities
       - **Party Performance Consistency**: High variance in party performance
       - **Isolation Forest ML**: Unsupervised learning to detect complex patterns

    4. **Method Comparison Framework**
       - Shows which methods agree on suspicious cases
       - Identifies high-consensus cases (3-4 methods agree)
       - Provides validation strength assessment

    ### How to Use These Results

    **High-Consensus Cases (3-4 methods):**
    - Strongest evidence from anomaly detection alone
    - Compare with probability analysis results for ultimate validation
    - Cases appearing in both analyses warrant detailed investigation

    **Moderate Cases (2 methods):**
    - Some unusual characteristics
    - Review in context of other information
    - May be edge cases or real anomalies

    **Next Steps:**
    Compare commission IDs from `high_consensus_cases.csv` with highly suspicious
    cases from the probability-based analysis to find commissions flagged by
    **both independent approaches** - these have the strongest overall evidence.
    """)
    return


if __name__ == "__main__":
    app.run()

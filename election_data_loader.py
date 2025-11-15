"""
Election Data Loading and Processing Utilities

This module provides reusable functions for loading and processing Czech election data
from CZSO (Czech Statistical Office). It handles data downloading, caching, and common
transformations used across multiple analysis notebooks.

Supports multiple election years for comparative analysis (2021, 2025, etc.)

Functions:
    - load_election_data(): Load main election results
    - load_municipality_data(): Load municipality (OBEC) information
    - load_party_data(): Load party information
    - calculate_party_statistics(): Calculate party vote totals and percentages
    - get_top_parties(): Get top N parties by vote share
    - calculate_municipality_sizes(): Calculate municipality sizes and categorize them
    - merge_municipality_sizes(): Merge municipality size data with election data
    - calculate_party_performance_by_obec(): Calculate party performance in each municipality
    - calculate_zero_vote_cases(): Find cases where top parties received 0 votes
    - create_commission_info(): Create commission-level aggregated information
"""

import pandas as pd
import requests
import zipfile
import io
from pathlib import Path
from typing import Tuple, List, Optional


# Configuration
DEFAULT_CACHE_DIR = Path(".")
DEFAULT_USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
DEFAULT_TIMEOUT = 30
DEFAULT_YEAR = 2025  # Default to most recent election


def get_data_urls(year: int) -> dict:
    """
    Get data URLs for a specific election year.

    Args:
        year: Election year (e.g., 2021, 2025)

    Returns:
        Dictionary with keys: 'election', 'municipality', 'party'
    """
    base_url = f"https://www.volby.cz/opendata/ps{year}/csv_od"
    return {
        'election': f"{base_url}/pst4p.zip",
        'municipality': f"{base_url}/pscoco.csv",
        'party': f"{base_url}/psrkl.csv"
    }


def load_election_data(
    year: int = DEFAULT_YEAR,
    cache_file: Optional[str] = None,
    force_download: bool = False
) -> pd.DataFrame:
    """
    Load election data from CZSO, with local caching.

    Downloads election results from the Czech Statistical Office and caches
    them locally as a Parquet file for faster subsequent loads.

    Args:
        year: Election year (default: 2025). Use 2021 for previous election.
        cache_file: Path to the cache file (default: "election_data_{year}.parquet")
        force_download: If True, download data even if cache exists

    Returns:
        DataFrame with election results containing columns like:
        - KSTRANA: Party code
        - ID_OKRSKY: Commission ID
        - OBEC: Municipality code
        - POC_HLASU: Number of votes

    Example:
        >>> # Load 2025 data
        >>> df_2025 = load_election_data(year=2025)
        >>> # Load 2021 data for comparison
        >>> df_2021 = load_election_data(year=2021)
        >>> print(f"Loaded {len(df_2025):,} rows from 2025")
    """
    # Default cache file includes year for easy comparison
    if cache_file is None:
        cache_file = f"election_data_{year}.parquet"

    parquet_file = Path(cache_file)
    urls = get_data_urls(year)

    # Check if cached data exists
    if parquet_file.exists() and not force_download:
        print(f"Loading cached {year} election data from {cache_file}")
        df = pd.read_parquet(parquet_file)
    else:
        print(f"Downloading {year} election data from CZSO...")
        headers = {'User-Agent': DEFAULT_USER_AGENT}
        response = requests.get(urls['election'], headers=headers, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()

        # Unzip and load the data
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
        csv_filename = zip_file.namelist()[0]

        with zip_file.open(csv_filename) as f:
            df = pd.read_csv(f)

        # Save to parquet for future use
        df.to_parquet(parquet_file)
        print(f"Data cached to {cache_file}")

    print(f"Loaded {len(df):,} election records from {year}")
    return df


def load_municipality_data(
    year: int = DEFAULT_YEAR,
    cache_file: Optional[str] = None,
    force_download: bool = False
) -> pd.DataFrame:
    """
    Load municipality (OBEC) data from CZSO, with local caching.

    Args:
        year: Election year (default: 2025)
        cache_file: Path to the cache file (default: "pscoco_{year}.csv")
        force_download: If True, download data even if cache exists

    Returns:
        DataFrame with municipality information containing columns like:
        - OBEC: Municipality code
        - NAZEVOBCE: Municipality name

    Example:
        >>> municipalities = load_municipality_data(year=2025)
        >>> print(f"Loaded {len(municipalities):,} municipalities")
    """
    # Default cache file includes year
    if cache_file is None:
        cache_file = f"pscoco_{year}.csv"

    cache_path = Path(cache_file)
    urls = get_data_urls(year)

    if not cache_path.exists() or force_download:
        print(f"Downloading {year} municipality data from CZSO...")
        headers = {'User-Agent': DEFAULT_USER_AGENT}
        response = requests.get(urls['municipality'], headers=headers, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()

        with open(cache_path, 'wb') as f:
            f.write(response.content)
        print(f"Municipality data cached to {cache_file}")
    else:
        print(f"Loading cached municipality data from {cache_file}")

    municipalities = pd.read_csv(cache_path, encoding='utf-8', dtype={'OBEC': str})
    print(f"Loaded {len(municipalities):,} municipalities")
    return municipalities


def load_party_data(
    year: int = DEFAULT_YEAR,
    cache_file: Optional[str] = None,
    force_download: bool = False
) -> pd.DataFrame:
    """
    Load party data from CZSO, with local caching.

    Args:
        year: Election year (default: 2025)
        cache_file: Path to the cache file (default: "psrkl_{year}.csv")
        force_download: If True, download data even if cache exists

    Returns:
        DataFrame with party information containing columns like:
        - KSTRANA: Party code
        - ZKRATKAK8: Party short name (8 chars)
        - NAZEVSTR: Party full name

    Example:
        >>> parties = load_party_data(year=2025)
        >>> print(f"Loaded {len(parties):,} parties")
    """
    # Default cache file includes year
    if cache_file is None:
        cache_file = f"psrkl_{year}.csv"

    cache_path = Path(cache_file)
    urls = get_data_urls(year)

    if not cache_path.exists() or force_download:
        print(f"Downloading {year} party data from CZSO...")
        headers = {'User-Agent': DEFAULT_USER_AGENT}
        response = requests.get(urls['party'], headers=headers, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()

        with open(cache_path, 'wb') as f:
            f.write(response.content)
        print(f"Party data cached to {cache_file}")
    else:
        print(f"Loading cached party data from {cache_file}")

    parties = pd.read_csv(cache_path, encoding='utf-8', dtype={'KSTRANA': str})
    print(f"Loaded {len(parties):,} parties")
    return parties


def calculate_party_statistics(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, float]:
    """
    Calculate overall party statistics from election data.

    Args:
        df: Election DataFrame with KSTRANA and POC_HLASU columns

    Returns:
        Tuple of (party_summary, party_totals, party_percentages, party_probabilities, total_votes):
        - party_summary: DataFrame with Total_Votes, Percentage, Probability columns
        - party_totals: Series of total votes per party
        - party_percentages: Series of vote percentages per party
        - party_probabilities: Series of vote probabilities per party (0-1)
        - total_votes: Total number of votes cast

    Example:
        >>> df = load_election_data()
        >>> summary, totals, pcts, probs, total = calculate_party_statistics(df)
        >>> print(summary.head())
    """
    # Calculate total votes per party
    party_totals = df.groupby('KSTRANA')['POC_HLASU'].sum().sort_values(ascending=False)

    # Get total number of votes cast
    total_votes = party_totals.sum()

    # Calculate percentage and probability for each party
    party_percentages = (party_totals / total_votes * 100).round(4)
    party_probabilities = party_totals / total_votes

    # Combine into a summary DataFrame
    party_summary = pd.DataFrame({
        'Total_Votes': party_totals,
        'Percentage': party_percentages,
        'Probability': party_probabilities
    })

    print(f"Calculated statistics for {len(party_summary)} parties")
    print(f"Total votes: {total_votes:,}")

    return party_summary, party_totals, party_percentages, party_probabilities, total_votes


def get_top_parties(party_summary: pd.DataFrame, n: int = 7) -> List[str]:
    """
    Get the top N parties by vote share.

    Args:
        party_summary: DataFrame from calculate_party_statistics()
        n: Number of top parties to return (default: 7)

    Returns:
        List of party codes for the top N parties

    Example:
        >>> summary, _, _, _, _ = calculate_party_statistics(df)
        >>> top_7 = get_top_parties(summary, n=7)
        >>> print(f"Top parties: {top_7}")
    """
    top_parties = party_summary.head(n).index.tolist()
    print(f"Top {n} parties: {top_parties}")
    return top_parties


def calculate_municipality_sizes(
    df: pd.DataFrame,
    n_categories: int = 5
) -> pd.DataFrame:
    """
    Calculate municipality sizes and categorize them.

    Args:
        df: Election DataFrame with OBEC, POC_HLASU, and ID_OKRSKY columns
        n_categories: Number of size categories to create (default: 5)

    Returns:
        DataFrame with columns:
        - OBEC: Municipality code
        - Total_Votes: Total votes in municipality
        - Num_Commissions: Number of commissions in municipality
        - Size_Category: Category (Very Small, Small, Medium, Large, Very Large)

    Example:
        >>> df = load_election_data()
        >>> muni_sizes = calculate_municipality_sizes(df)
        >>> print(muni_sizes.head())
    """
    # Calculate municipality sizes based on total votes
    municipality_sizes = df.groupby('OBEC').agg({
        'POC_HLASU': 'sum',
        'ID_OKRSKY': 'nunique'
    }).reset_index()
    municipality_sizes.columns = ['OBEC', 'Total_Votes', 'Num_Commissions']

    # Categorize municipalities by size using quantiles
    if n_categories == 5:
        labels = ['Very Small', 'Small', 'Medium', 'Large', 'Very Large']
    else:
        labels = [f'Category_{i+1}' for i in range(n_categories)]

    municipality_sizes['Size_Category'] = pd.qcut(
        municipality_sizes['Total_Votes'],
        q=n_categories,
        labels=labels,
        duplicates='drop'
    )

    print(f"Calculated sizes for {len(municipality_sizes)} municipalities")
    print(f"\nMunicipality size distribution:")
    print(municipality_sizes.groupby('Size_Category', observed=False)['OBEC'].count())

    return municipality_sizes


def merge_municipality_sizes(
    df: pd.DataFrame,
    municipality_sizes: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge municipality size information back to the main election DataFrame.

    Args:
        df: Election DataFrame
        municipality_sizes: DataFrame from calculate_municipality_sizes()

    Returns:
        DataFrame with municipality size columns added

    Example:
        >>> df = load_election_data()
        >>> muni_sizes = calculate_municipality_sizes(df)
        >>> df_with_sizes = merge_municipality_sizes(df, muni_sizes)
    """
    df_with_size = df.merge(
        municipality_sizes[['OBEC', 'Total_Votes', 'Size_Category']],
        on='OBEC',
        suffixes=('', '_Municipality')
    )

    print(f"Merged municipality sizes: {len(df_with_size):,} records")
    return df_with_size


def calculate_party_performance_by_obec(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate party performance statistics for each municipality (OBEC).

    This is crucial for OBEC-specific probability calculations, as parties
    perform differently in different municipalities.

    Args:
        df: Election DataFrame with KSTRANA, OBEC, and POC_HLASU columns

    Returns:
        DataFrame with columns:
        - Party: Party code
        - OBEC: Municipality code
        - Total_Votes: Total votes for party in municipality
        - Num_Commissions: Number of commissions where party appeared
        - OBEC_Total_Votes: Total votes in municipality
        - Vote_Share_In_OBEC: Party's vote share in municipality (%)
        - Probability_In_OBEC: Party's vote probability in municipality (0-1)

    Example:
        >>> df = load_election_data()
        >>> party_by_obec = calculate_party_performance_by_obec(df)
        >>> print(party_by_obec.head())
    """
    # Calculate party performance by OBEC
    party_by_obec = df.groupby(['KSTRANA', 'OBEC'])['POC_HLASU'].agg([
        'sum', 'count'
    ]).reset_index()
    party_by_obec.columns = ['Party', 'OBEC', 'Total_Votes', 'Num_Commissions']

    # Calculate total votes per OBEC
    obec_totals = df.groupby('OBEC')['POC_HLASU'].sum().reset_index()
    obec_totals.columns = ['OBEC', 'OBEC_Total_Votes']

    # Merge and calculate vote share
    party_by_obec = party_by_obec.merge(obec_totals, on='OBEC')
    party_by_obec['Vote_Share_In_OBEC'] = (
        party_by_obec['Total_Votes'] / party_by_obec['OBEC_Total_Votes'] * 100
    )
    party_by_obec['Probability_In_OBEC'] = party_by_obec['Vote_Share_In_OBEC'] / 100

    print(f"Calculated party performance in {party_by_obec['OBEC'].nunique()} municipalities")
    print(f"Total party-municipality combinations: {len(party_by_obec):,}")

    return party_by_obec


def create_commission_info(df: pd.DataFrame, include_size_category: bool = False) -> pd.DataFrame:
    """
    Create commission-level aggregated information.

    Args:
        df: Election DataFrame with ID_OKRSKY, POC_HLASU, and OBEC columns
        include_size_category: If True, include Size_Category column from df

    Returns:
        DataFrame with commission-level information:
        - Commission_ID (or ID_OKRSKY): Commission identifier
        - Total_Votes_In_Commission: Total votes in commission
        - OBEC: Municipality code
        - Size_Category: Municipality size category (if include_size_category=True)

    Example:
        >>> df = load_election_data()
        >>> comm_info = create_commission_info(df)
    """
    agg_dict = {
        'POC_HLASU': 'sum',
        'OBEC': 'first'
    }

    if include_size_category and 'Size_Category' in df.columns:
        agg_dict['Size_Category'] = 'first'
        if 'Total_Votes' in df.columns:
            agg_dict['Total_Votes'] = 'first'

    commission_info = df.groupby('ID_OKRSKY').agg(agg_dict).reset_index()

    # Rename columns
    col_names = ['Commission_ID', 'Total_Votes_In_Commission', 'OBEC']
    if include_size_category and 'Size_Category' in df.columns:
        col_names.append('Municipality_Size_Category')
        if 'Total_Votes' in df.columns:
            col_names.append('Municipality_Total_Votes')

    commission_info.columns = col_names

    print(f"Created commission info for {len(commission_info):,} commissions")
    return commission_info


def calculate_zero_vote_cases(
    df: pd.DataFrame,
    top_parties: List[str],
    commission_info: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Find all cases where top parties received 0 votes in commissions.

    Args:
        df: Election DataFrame
        top_parties: List of party codes to analyze
        commission_info: Optional pre-computed commission info DataFrame

    Returns:
        DataFrame with zero-vote cases:
        - Party: Party code
        - Commission_ID: Commission where party got 0 votes
        - Total_Votes_In_Commission: Total votes in that commission
        - OBEC: Municipality code

    Example:
        >>> df = load_election_data()
        >>> summary, _, _, _, _ = calculate_party_statistics(df)
        >>> top_parties = get_top_parties(summary, n=7)
        >>> zero_cases = calculate_zero_vote_cases(df, top_parties)
    """
    # Get all unique commissions
    all_commissions = df['ID_OKRSKY'].unique()

    # Create all possible combinations of top parties and commissions
    all_combinations = pd.MultiIndex.from_product(
        [top_parties, all_commissions],
        names=['Party', 'Commission_ID']
    ).to_frame(index=False)

    # Get actual combinations present in the data
    actual_combinations = df[df['KSTRANA'].isin(top_parties)][
        ['KSTRANA', 'ID_OKRSKY']
    ].copy()
    actual_combinations.columns = ['Party', 'Commission_ID']
    actual_combinations['Present'] = True

    # Merge to find missing combinations (where parties got 0 votes)
    combined = all_combinations.merge(
        actual_combinations,
        on=['Party', 'Commission_ID'],
        how='left'
    )
    zero_votes = combined[combined['Present'].isna()][['Party', 'Commission_ID']].copy()

    # Create or use provided commission info
    if commission_info is None:
        commission_info = create_commission_info(df, include_size_category=False)

    # Merge to get commission details
    zero_votes_df = zero_votes.merge(commission_info, on='Commission_ID')

    print(f"Found {len(zero_votes_df):,} zero-vote cases for {len(top_parties)} parties")

    return zero_votes_df


def merge_obec_probabilities(
    zero_votes_df: pd.DataFrame,
    party_by_obec: pd.DataFrame,
    party_summary: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge OBEC-specific probabilities and calculate zero-vote probabilities.

    Args:
        zero_votes_df: DataFrame from calculate_zero_vote_cases()
        party_by_obec: DataFrame from calculate_party_performance_by_obec()
        party_summary: DataFrame from calculate_party_statistics()

    Returns:
        DataFrame with probability calculations:
        - All columns from zero_votes_df
        - Vote_Share_In_OBEC: Party's vote share in municipality
        - Probability_In_OBEC: Party's probability in municipality
        - Party_Percentage: Party's national vote percentage
        - Party_Probability_Overall: Party's national probability
        - Probability_of_Zero_Overall: P(0 votes) using national probability
        - Probability_of_Zero_OBEC: P(0 votes) using OBEC-specific probability
        - Probability_of_Zero: Best estimate (OBEC if available, else overall)
        - Used_OBEC_Probability: Boolean flag
        - Probability_of_Zero_Percent: Probability as percentage

    Example:
        >>> zero_cases = calculate_zero_vote_cases(df, top_parties)
        >>> party_by_obec = calculate_party_performance_by_obec(df)
        >>> summary, _, _, _, _ = calculate_party_statistics(df)
        >>> prob_analysis = merge_obec_probabilities(zero_cases, party_by_obec, summary)
    """
    # Merge OBEC-specific probabilities
    prob_analysis = zero_votes_df.merge(
        party_by_obec[['Party', 'OBEC', 'Vote_Share_In_OBEC', 'Probability_In_OBEC']],
        on=['Party', 'OBEC'],
        how='left'
    )

    # Add overall party statistics
    prob_analysis['Party_Probability_Overall'] = prob_analysis['Party'].map(
        lambda x: party_summary.loc[x, 'Probability'] if x in party_summary.index else None
    )
    prob_analysis['Party_Percentage'] = prob_analysis['Party'].map(
        lambda x: party_summary.loc[x, 'Percentage'] if x in party_summary.index else None
    )

    # Calculate probability using overall (national) statistics
    prob_analysis['Probability_of_Zero_Overall'] = (
        1 - prob_analysis['Party_Probability_Overall']
    ) ** prob_analysis['Total_Votes_In_Commission']

    # Calculate OBEC-specific probability (more accurate)
    prob_analysis['Probability_of_Zero_OBEC'] = (
        1 - prob_analysis['Probability_In_OBEC']
    ) ** prob_analysis['Total_Votes_In_Commission']

    # Use OBEC-specific where available, otherwise fall back to overall
    prob_analysis['Probability_of_Zero'] = prob_analysis['Probability_of_Zero_OBEC'].fillna(
        prob_analysis['Probability_of_Zero_Overall']
    )

    # Flag whether we used OBEC-specific probability
    prob_analysis['Used_OBEC_Probability'] = prob_analysis['Probability_In_OBEC'].notna()

    # Convert to percentage
    prob_analysis['Probability_of_Zero_Percent'] = prob_analysis['Probability_of_Zero'] * 100

    print(f"Merged probabilities for {len(prob_analysis):,} zero-vote cases")
    print(f"Cases with OBEC-specific data: {prob_analysis['Used_OBEC_Probability'].sum():,}")

    return prob_analysis


def add_suspiciousness_flags(
    prob_analysis: pd.DataFrame,
    suspicious_threshold: float = 0.01,
    highly_suspicious_threshold: float = 0.001,
    require_obec_data: bool = True
) -> pd.DataFrame:
    """
    Add suspiciousness flags to probability analysis.

    Args:
        prob_analysis: DataFrame from merge_obec_probabilities()
        suspicious_threshold: Probability threshold for suspicious cases (default: 0.01 = 1%)
        highly_suspicious_threshold: Probability threshold for highly suspicious (default: 0.001 = 0.1%)
        require_obec_data: If True, only flag cases with OBEC-specific data

    Returns:
        DataFrame with added columns:
        - Is_Suspicious: Boolean flag for P < suspicious_threshold
        - Is_Highly_Suspicious: Boolean flag for P < highly_suspicious_threshold

    Example:
        >>> prob_analysis = merge_obec_probabilities(zero_cases, party_by_obec, summary)
        >>> prob_analysis = add_suspiciousness_flags(prob_analysis)
        >>> print(f"Suspicious cases: {prob_analysis['Is_Suspicious'].sum()}")
    """
    result = prob_analysis.copy()

    if require_obec_data:
        result['Is_Suspicious'] = (
            (result['Probability_of_Zero'] < suspicious_threshold) &
            result['Used_OBEC_Probability']
        )
        result['Is_Highly_Suspicious'] = (
            (result['Probability_of_Zero'] < highly_suspicious_threshold) &
            result['Used_OBEC_Probability']
        )
    else:
        result['Is_Suspicious'] = result['Probability_of_Zero'] < suspicious_threshold
        result['Is_Highly_Suspicious'] = result['Probability_of_Zero'] < highly_suspicious_threshold

    print(f"Suspicious cases (< {suspicious_threshold*100}%): {result['Is_Suspicious'].sum()}")
    print(f"Highly suspicious (< {highly_suspicious_threshold*100}%): {result['Is_Highly_Suspicious'].sum()}")

    return result

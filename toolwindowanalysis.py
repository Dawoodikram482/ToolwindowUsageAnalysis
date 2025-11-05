import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv('data/toolwindow_data.csv')

print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head(10))
print("\nData types:")
print(df.dtypes)
print("\nBasic statistics:")
print(df.describe())
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nEvent types:")
print(df['event'].value_counts())
print("\nOpen types (non-null):")
print(df['open_type'].value_counts())


def process_toolwindow_data(df):
    """
    Process tool window events to extract open-close pairs and durations.

    Assumptions:
    - Events are processed in chronological order per user
    - Multiple consecutive 'opened' events: previous session implicitly closed (incomplete)
    - 'closed' without 'opened': orphaned event, ignored
    - 'opened' without 'closed' by end: incomplete session, excluded
    """

    # Sort by user and timestamp
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    sessions = []
    open_stack = {}  # Track currently open sessions per user

    # Statistics for data quality
    stats_dict = {
        'total_events': len(df),
        'opened_events': 0,
        'closed_events': 0,
        'orphaned_closes': 0,
        'implicit_closes': 0,
        'matched_sessions': 0,
        'incomplete_opens': 0
    }

    for idx, row in df.iterrows():
        user_id = row['user_id']
        event = row['event']
        timestamp = row['timestamp']
        open_type = row['open_type']

        if event == 'opened':
            stats_dict['opened_events'] += 1

            # If there's already an open for this user, implicitly close it
            if user_id in open_stack:
                stats_dict['implicit_closes'] += 1
                # Don't record this incomplete session

            # Record new open event
            open_stack[user_id] = {
                'open_time': timestamp,
                'open_type': open_type
            }

        elif event == 'closed':
            stats_dict['closed_events'] += 1

            # Match with open event
            if user_id in open_stack:
                open_info = open_stack[user_id]
                duration = timestamp - open_info['open_time']

                # Only record positive durations
                if duration > 0:
                    sessions.append({
                        'user_id': user_id,
                        'open_time': open_info['open_time'],
                        'close_time': timestamp,
                        'duration_ms': duration,
                        'open_type': open_info['open_type']
                    })
                    stats_dict['matched_sessions'] += 1

                # Remove from stack
                del open_stack[user_id]
            else:
                # Close without open - orphaned event
                stats_dict['orphaned_closes'] += 1

    # Count incomplete opens (still in stack at end)
    stats_dict['incomplete_opens'] = len(open_stack)

    print("\n" + "="*60)
    print("DATA PROCESSING STATISTICS")
    print("="*60)
    for key, value in stats_dict.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    return pd.DataFrame(sessions), stats_dict


def analyze_durations(sessions_df):
    """
    Perform comprehensive statistical analysis on session durations.
    """

    # Convert durations to more interpretable units
    sessions_df['duration_sec'] = sessions_df['duration_ms'] / 1000
    sessions_df['duration_min'] = sessions_df['duration_sec'] / 60

    # Separate manual and auto sessions
    manual = sessions_df[sessions_df['open_type'] == 'manual']['duration_sec']
    auto = sessions_df[sessions_df['open_type'] == 'auto']['duration_sec']

    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    # Detailed summary statistics
    summary = sessions_df.groupby('open_type')['duration_sec'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('q25', lambda x: x.quantile(0.25)),
        ('q75', lambda x: x.quantile(0.75)),
        ('q90', lambda x: x.quantile(0.90)),
        ('q95', lambda x: x.quantile(0.95))
    ]).round(2)

    print("\nDuration Statistics (in seconds):")
    print(summary)

    # Print in minutes for context
    print("\n\nKey Metrics (in minutes):")
    for open_type in ['manual', 'auto']:
        data = sessions_df[sessions_df['open_type'] == open_type]['duration_min']
        print(f"\n{open_type.upper()}:")
        print(f"  Mean: {data.mean():.2f} min")
        print(f"  Median: {data.median():.2f} min")
        print(f"  Std Dev: {data.std():.2f} min")

    # Statistical tests
    print("\n" + "="*60)
    print("STATISTICAL TESTS")
    print("="*60)

    # 1. Shapiro-Wilk test for normality
    _, p_manual_norm = stats.shapiro(manual.sample(min(5000, len(manual))))
    _, p_auto_norm = stats.shapiro(auto.sample(min(5000, len(auto))))

    print(f"\nShapiro-Wilk Normality Test:")
    print(f"  Manual p-value: {p_manual_norm:.6f} - {'Normal' if p_manual_norm > 0.05 else 'Not Normal'}")
    print(f"  Auto p-value: {p_auto_norm:.6f} - {'Normal' if p_auto_norm > 0.05 else 'Not Normal'}")

    # 2. Mann-Whitney U test (non-parametric)
    statistic, p_value = stats.mannwhitneyu(manual, auto, alternative='two-sided')
    print(f"\nMann-Whitney U Test (Non-Parametric):")
    print(f"  Test statistic: {statistic:.4f}")
    print(f"  P-value: {p_value:.6f}")
    print(f"  Result: {'Statistically significant' if p_value < 0.05 else 'Not statistically significant'} at α=0.05")

    if p_value < 0.05:
        winner = "Manual" if manual.median() > auto.median() else "Auto"
        print(f"  → {winner} opens have significantly different durations")

    # 3. Independent T-test (parametric)
    t_stat, t_pvalue = stats.ttest_ind(manual, auto)
    print(f"\nIndependent T-Test (Parametric):")
    print(f"  Test statistic: {t_stat:.4f}")
    print(f"  P-value: {t_pvalue:.6f}")

    # 4. Cohen's d for effect size
    pooled_std = np.sqrt(((len(manual)-1)*manual.std()**2 + (len(auto)-1)*auto.std()**2) / (len(manual)+len(auto)-2))
    cohens_d = (manual.mean() - auto.mean()) / pooled_std

    print(f"\nEffect Size (Cohen's d): {cohens_d:.4f}")
    if abs(cohens_d) < 0.2:
        effect_interpretation = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_interpretation = "small"
    elif abs(cohens_d) < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"
    print(f"  Interpretation: {effect_interpretation} effect")

    # 5. Confidence intervals for means
    print(f"\n95% Confidence Intervals for Mean Duration (seconds):")
    for open_type, data in [('manual', manual), ('auto', auto)]:
        ci = stats.t.interval(0.95, len(data)-1, loc=data.mean(), scale=stats.sem(data))
        print(f"  {open_type.capitalize()}: ({ci[0]:.2f}, {ci[1]:.2f})")

    # 6. Bootstrapped confidence intervals for medians
    print(f"\n95% Bootstrapped Confidence Intervals for Median Duration (seconds):")
    print(f"  (Using 10,000 bootstrap resamples)")

    def bootstrap_median_ci(data, n_bootstrap=10000, confidence=0.95):
        """Calculate bootstrapped confidence interval for median"""
        np.random.seed(42)  # For reproducibility
        bootstrap_medians = []

        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_medians.append(np.median(sample))

        alpha = 1 - confidence
        lower = np.percentile(bootstrap_medians, (alpha/2) * 100)
        upper = np.percentile(bootstrap_medians, (1 - alpha/2) * 100)

        return lower, upper, bootstrap_medians

    for open_type, data in [('manual', manual), ('auto', auto)]:
        lower, upper, boot_medians = bootstrap_median_ci(data)
        print(f"  {open_type.capitalize()}: ({lower:.2f}, {upper:.2f}) seconds")
        print(f"    → Median: {data.median():.2f} seconds")

    # Calculate the ratio and its confidence
    manual_lower, manual_upper, manual_boots = bootstrap_median_ci(manual)
    auto_lower, auto_upper, auto_boots = bootstrap_median_ci(auto)

    # Bootstrap the ratio itself
    ratio_boots = np.array(auto_boots) / np.array(manual_boots)
    ratio_lower = np.percentile(ratio_boots, 2.5)
    ratio_upper = np.percentile(ratio_boots, 97.5)
    actual_ratio = auto.median() / manual.median()

    print(f"\n  Ratio (Auto/Manual): {actual_ratio:.2f}x")
    print(f"    95% CI for ratio: ({ratio_lower:.2f}x, {ratio_upper:.2f}x)")
    print(f"    → Auto opens stay open {actual_ratio:.1f}x longer (median comparison)")

    return sessions_df, summary


def create_visualizations(sessions_df):
    """
    Create comprehensive visualizations for the analysis.
    """

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Box plot
    sessions_df.boxplot(column='duration_sec', by='open_type', ax=axes[0, 0])
    axes[0, 0].set_title('Duration by Open Type (Boxplot)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Open Type', fontsize=10)
    axes[0, 0].set_ylabel('Duration (seconds)', fontsize=10)
    axes[0, 0].get_figure().suptitle('')  # Remove default title

    # 2. Histogram with KDE
    manual_dur = sessions_df[sessions_df['open_type'] == 'manual']['duration_sec']
    auto_dur = sessions_df[sessions_df['open_type'] == 'auto']['duration_sec']

    axes[0, 1].hist([manual_dur, auto_dur], bins=50, label=['Manual', 'Auto'],
                    alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Duration (seconds)', fontsize=10)
    axes[0, 1].set_ylabel('Frequency', fontsize=10)
    axes[0, 1].set_title('Distribution of Durations', fontsize=12, fontweight='bold')
    axes[0, 1].legend()

    # 3. Violin plot
    sns.violinplot(data=sessions_df, x='open_type', y='duration_sec', ax=axes[0, 2])
    axes[0, 2].set_title('Duration Distribution (Violin Plot)', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Open Type', fontsize=10)
    axes[0, 2].set_ylabel('Duration (seconds)', fontsize=10)

    # 4. Log scale histogram to see distribution better
    axes[1, 0].hist([np.log10(manual_dur + 1), np.log10(auto_dur + 1)],
                    bins=50, label=['Manual', 'Auto'], alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Log10(Duration + 1) seconds', fontsize=10)
    axes[1, 0].set_ylabel('Frequency', fontsize=10)
    axes[1, 0].set_title('Log-Scale Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].legend()

    # 5. CDF plot
    manual_sorted = np.sort(manual_dur)
    auto_sorted = np.sort(auto_dur)
    manual_cdf = np.arange(1, len(manual_sorted) + 1) / len(manual_sorted)
    auto_cdf = np.arange(1, len(auto_sorted) + 1) / len(auto_sorted)

    axes[1, 1].plot(manual_sorted, manual_cdf, label='Manual', linewidth=2)
    axes[1, 1].plot(auto_sorted, auto_cdf, label='Auto', linewidth=2)
    axes[1, 1].set_xlabel('Duration (seconds)', fontsize=10)
    axes[1, 1].set_ylabel('Cumulative Probability', fontsize=10)
    axes[1, 1].set_title('Cumulative Distribution Function', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Summary statistics comparison
    summary_data = sessions_df.groupby('open_type')['duration_sec'].agg(['mean', 'median'])
    x_pos = np.arange(len(summary_data.index))
    width = 0.35

    axes[1, 2].bar(x_pos - width/2, summary_data['mean'], width, label='Mean', alpha=0.8)
    axes[1, 2].bar(x_pos + width/2, summary_data['median'], width, label='Median', alpha=0.8)
    axes[1, 2].set_xlabel('Open Type', fontsize=10)
    axes[1, 2].set_ylabel('Duration (seconds)', fontsize=10)
    axes[1, 2].set_title('Mean vs Median Comparison', fontsize=12, fontweight='bold')
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(summary_data.index)
    axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig('toolwindow_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved as 'toolwindow_analysis.png'")
    plt.close()



print("\n\n" + "="*60)
print("STARTING TOOLWINDOW ANALYSIS")
print("="*60)

# Process the data
sessions_df, processing_stats = process_toolwindow_data(df)

print(f"\n{'='*60}")
print(f"PROCESSED SESSIONS SUMMARY")
print(f"{'='*60}")
print(f"Total matched sessions: {len(sessions_df)}")
print(f"\nSessions by open type:")
print(sessions_df['open_type'].value_counts())
print(f"\nSessions per user (top 10):")
print(sessions_df['user_id'].value_counts().head(10))

# Analyze durations
sessions_df, summary = analyze_durations(sessions_df)

# Create visualizations
print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)
create_visualizations(sessions_df)

# Save results
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)
sessions_df.to_csv('processed_sessions.csv', index=False)
summary.to_csv('summary_statistics.csv')





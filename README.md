# Toolwindow Usage Analysis

## Overview
Analyzes IDE toolwindow usage data to determine if there's a significant difference in how long toolwindows stay open depending on whether they were opened manually or automatically.

## Quick Setup & Execution Guide

## Prerequisites
- Python 3.8 or higher installed
- pip package manager

## Setup (One-time)

1. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or manually:
   ```bash
   pip install pandas numpy matplotlib seaborn scipy
   ```

## Running the Analysis

1. **Ensure your dataset is in place:**
   - File should be at: `data/toolwindow_data.csv`
   - Should contain columns: timestamp, event, open_type, user_id

2. **Run the analysis script:**
   ```bash
   python toolwindowanalysis.py
   ```

3. **Wait for completion:**
   - The script will print progress to console
   - Takes approximately 30-60 seconds to complete
   - Generates visualizations automatically

## Expected Output Files

After successful execution, you'll have:

1. **processed_sessions.csv**
2. **summary_statistics.csv**
3. **toolwindow_analysis.png** 
## Console Output

The script provides detailed output including:
- Initial data exploration (shape, types, missing values)
- Data processing statistics (matched sessions, orphaned events, etc.)
- Summary statistics (mean, median, std dev by group)
- Statistical test results (Mann-Whitney U, t-test, Cohen's d)
- 95% confidence intervals

## Troubleshooting

### Issue: "No module named 'pandas'" (or similar)
**Solution**: Install requirements: `pip install -r requirements.txt`

### Issue: "FileNotFoundError: data/toolwindow_data.csv"
**Solution**: Ensure dataset is in `data/` folder with correct filename

### Issue: Script runs but no output files
**Solution**: Check file permissions in the directory

### Issue: Visualization doesn't appear
**Solution**: Script saves PNG file automatically, check for `toolwindow_analysis.png`


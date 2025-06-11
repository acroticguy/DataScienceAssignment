# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Areios Pagos Decision Analysis: Extraction and Visualization
#
# This notebook performs two main tasks:
# 1.  **Data Extraction:** It scrapes legal decision metadata and cited articles from the Areios Pagos (Supreme Court of Greece) website for specified sectors and years.
# 2.  **Data Analysis & Visualization:** It loads the extracted data, cleans it, and generates various plots to explore trends, distributions, and patterns in the decisions.

# ## 1. Setup and Imports
#
# First, we import all the necessary libraries for web scraping, data manipulation, and plotting. We also configure warnings to keep the output clean.

# +
import requests
from bs4 import BeautifulSoup # To parse the HTML response
import re
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast  # For safely evaluating string representations of lists
from collections import Counter
import warnings

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='seaborn')
warnings.filterwarnings('ignore', category=FutureWarning)

print("Libraries imported successfully.")
# -

# ## 2. Configuration
#
# Define constants and parameters used throughout the notebook, such as base URLs, target years/sectors for scraping, the output filename, and plotting preferences.

# +
# --- Scraping Configuration ---
BASE_URL = "https://areiospagos.gr/nomologia/"
# Define target sectors and years (using codes from website analysis)
# '1': Politiko (Civil), '2': Poiniko (Criminal)
TARGET_SECTORS = {'1': 'Politiko', '2': 'Poiniko'}
TARGET_YEAR = '2024' # The year to scrape data for
OUTPUT_CSV_FILENAME = 'output_data.csv' # File to save scraped data
REQUEST_DELAY = 0.2 # Seconds to wait between requests to be polite to the server
REQUEST_TIMEOUT = 20 # Seconds to wait for server response

# --- Analysis Configuration ---
TOP_N_ARTICLES = 15  # How many top articles to show in plots
FIG_SIZE = (12, 6)   # Default figure size for plots

# --- Article Code Mapping ---
# Maps internal keys to the column names expected in the DataFrame
ARTICLE_COLUMNS = {
    'AK': 'articles_ak',
    'KPolΔ': 'articles_kpolΔ',
    'PK': 'articles_pk',
    'KPD': 'articles_kpd'
}

print("Configuration set.")
# -

# ## 3. Data Extraction (Web Scraping)
#
# This section contains the functions responsible for fetching and parsing data from the Areios Pagos website.
#
# **Warning:** Web scraping can be fragile. Changes to the website's structure may break this code. Always scrape responsibly by using delays between requests.

# ### 3.1 Helper Function: Extract Data from Decision Page
#
# This function takes the HTML content (as a BeautifulSoup object) of a single decision page and extracts relevant details like judges, text blocks, and cited articles using string searching and regular expressions.

# +
def extract_decision_data(soup):
    """
    Extracts specific data points from the BeautifulSoup object of a decision page.
    """
    data = {
        "judges": None,
        "court_intro_text": None,
        "reasoning_text": None,
        "decision_text": None,
        "articles_ak": set(),      # Use sets to store unique articles
        "articles_kpolΔ": set(),
        "articles_pk": set(),
        "articles_kpd": set(),
    }

    try:
        # Find the main content area (adjust selector if needed based on other decisions)
        # The <font face="Arial" size="3"> tag seems to contain the main legal text
        content_font = soup.find('font', {'face': 'Arial', 'size': '3'})
        if not content_font:
            print("Warning: Main content font tag not found on a decision page.")
            return data # Return empty data if content not found

        # Get text with line breaks preserved, important for finding specific lines/paragraphs
        # Using '\n' as separator helps split later. Strip removes leading/trailing whitespace from the whole block.
        full_text = content_font.get_text(separator='\n', strip=True)
        lines = full_text.split('\n') # Split into lines for easier processing

        # --- Extract Judges ---
        judges_prefix = "Συγκροτήθηκε από τους δικαστές"
        for line in lines:
            clean_line = line.strip()
            if clean_line.startswith(judges_prefix):
                data["judges"] = clean_line.replace(judges_prefix, '').strip().rstrip(':') # Clean up
                break # Found it, no need to check further

        # --- Extract Text Blocks using Markers ---
        marker1 = "ΤΟ ΔΙΚΑΣΤΗΡΙΟ ΤΟΥ ΑΡΕΙΟΥ ΠΑΓΟΥ"
        marker2 = "ΣΚΕΦΘΗΚΕ ΣΥΜΦΩΝΑ ΜΕ ΤΟ ΝΟΜΟ"
        marker3 = "ΓΙΑ ΤΟΥΣ ΛΟΓΟΥΣ ΑΥΤΟΥΣ"
        end_decision_marker = "ΚΡΙΘΗΚΕ," # Often marks the end of the decision block

        try:
            # Find indices of markers in the full text
            start_intro = full_text.find(marker1)
            start_reasoning = full_text.find(marker2)
            start_decision = full_text.find(marker3)
            end_decision = full_text.find(end_decision_marker, start_decision if start_decision != -1 else 0)

            if start_intro != -1 and start_reasoning != -1:
                data["court_intro_text"] = full_text[start_intro + len(marker1):start_reasoning].strip()
            elif start_intro != -1: # Fallback if reasoning marker missing
                 data["court_intro_text"] = full_text[start_intro + len(marker1):].strip() # Take rest of text

            if start_reasoning != -1 and start_decision != -1:
                data["reasoning_text"] = full_text[start_reasoning + len(marker2):start_decision].strip()
            elif start_reasoning != -1: # Fallback if decision marker missing
                data["reasoning_text"] = full_text[start_reasoning + len(marker2):].strip() # Take rest of text


            if start_decision != -1:
                 if end_decision != -1:
                     data["decision_text"] = full_text[start_decision:end_decision].strip()
                 else:
                     # Fallback: take text from marker3 onwards if end marker not found
                     data["decision_text"] = full_text[start_decision:].strip()


        except Exception as e:
            print(f"Warning: Error extracting text blocks: {e}")


        # --- Extract Articles using Regex ---
        # We'll search primarily in the reasoning and decision text, as they contain the legal basis
        search_text = (data.get("reasoning_text") or "") + "\n" + (data.get("decision_text") or "")

        # Define regex patterns for each code - more robust patterns
        # Pattern format: (Code Name) (Article Num) [Optional: (Specifier) (Specifier Num/Letter)]
        # Or: (Article Num) (Specifier like 'του') (Code Name) [Optional: (Specifier) (Specifier Num/Letter)]
        # Added flexibility for variations like Α.Κ., Κ.Πολ.Δ. etc. and common specifiers.
        patterns = {
            "articles_kpolΔ": r'(?:(ΚΠολΔ|Κ\.Π[οό]λ\.Δ\.?)\s+(\d+)|(\d+)\s+(?:του\s+)?(?:ΚΠολΔ|Κ\.Π[οό]λ\.Δ\.?))(?:\s+(παρ|αριθμ|αρ|εδ|στοιχ|περ)\.?)?\s*([\d]+|[α-ζάέήίόύώ]+|[Α-ΩΆΈΉΊΌΎΏ]+)?',
            "articles_ak":    r'(?:(ΑΚ|Α\.Κ\.?)\s+(\d+)|(\d+)\s+(?:του\s+)?(?:ΑΚ|Α\.Κ\.?))(?:\s+(παρ|αριθμ|αρ|εδ|στοιχ|περ)\.?)?\s*([\d]+|[α-ζάέήίόύώ]+|[Α-ΩΆΈΉΊΌΎΏ]+)?',
            "articles_pk":    r'(?:(ΠΚ|Π\.Κ\.?)\s+(\d+)|(\d+)\s+(?:του\s+)?(?:ΠΚ|Π\.Κ\.?))(?:\s+(παρ|αριθμ|αρ|εδ|στοιχ|περ)\.?)?\s*([\d]+|[α-ζάέήίόύώ]+|[Α-ΩΆΈΉΊΌΎΏ]+)?',
            "articles_kpd":   r'(?:(ΚΠΔ|Κ\.Π\.Δ\.?)\s+(\d+)|(\d+)\s+(?:του\s+)?(?:ΚΠΔ|Κ\.Π\.Δ\.?))(?:\s+(παρ|αριθμ|αρ|εδ|στοιχ|περ)\.?)?\s*([\d]+|[α-ζάέήίόύώ]+|[Α-ΩΆΈΉΊΌΎΏ]+)?',
        }

        for key, pattern in patterns.items():
            # Use finditer to get match objects, which helps avoid overlapping matches if patterns are complex
            for match in re.finditer(pattern, search_text, re.IGNORECASE | re.UNICODE):
                # Determine article number (it's either group 2 or 3 based on pattern structure)
                article_num = match.group(2) or match.group(4) # Adjusted group indices based on updated regex
                # Specifier (group 5) and its value (group 6) are optional
                specifier = match.group(5)
                specifier_val = match.group(6)

                if article_num: # Ensure we found an article number
                    formatted_article = str(article_num)
                    if specifier:
                        # Clean specifier (remove trailing dot if present)
                        clean_specifier = specifier.rstrip('.').lower()
                        # Standardize common specifiers
                        if clean_specifier in ['αρ', 'αριθμ']: clean_specifier = 'αρ.'
                        elif clean_specifier in ['εδ']: clean_specifier = 'εδ.'
                        elif clean_specifier in ['στοιχ']: clean_specifier = 'στοιχ.'
                        elif clean_specifier in ['περ']: clean_specifier = 'περ.'
                        elif clean_specifier in ['παρ']: clean_specifier = 'παρ.'

                        formatted_article += f" {clean_specifier}"
                        if specifier_val:
                            formatted_article += f" {specifier_val}"
                    # Add the unique formatted article string to the correct set
                    data[key].add(formatted_article.strip())

    except Exception as e:
        print(f"Error processing decision details: {e}")
        # Optionally re-raise or log the error more formally

    # Convert sets back to sorted lists for consistency
    data["articles_ak"] = sorted(list(data["articles_ak"]))
    data["articles_kpolΔ"] = sorted(list(data["articles_kpolΔ"]))
    data["articles_pk"] = sorted(list(data["articles_pk"]))
    data["articles_kpd"] = sorted(list(data["articles_kpd"]))

    return data

print("Helper function 'extract_decision_data' defined.")
# -

# ### 3.2 Main Scraping Function: Crawl Sector for a Year
#
# This function orchestrates the scraping process for a specific sector (e.g., Civil, Criminal) and year. It handles session management, POST requests to the search form, parsing the results page, and then iterating through individual decision links to extract data using the `extract_decision_data` helper.

# +
def crawl_for_sector(sector_code: str, sector_name: str, year: str):
    """
    Crawls the Areios Pagos website for decisions of a specific sector and year.

    Args:
        sector_code (str): The internal code used by the website for the sector (e.g., '1', '2').
        sector_name (str): The human-readable name of the sector (e.g., 'Politiko').
        year (str): The target year (e.g., '2024').

    Returns:
        list: A list of dictionaries, where each dictionary contains data for one decision.
    """
    print(f"\n--- Starting crawl for Sector: {sector_name} ({sector_code}), Year: {year} ---")
    # We use a Session object to handle cookies automatically
    session = requests.Session()

    search_results_url = BASE_URL + "apofaseis_result.asp?S=1"
    initial_form_page_url = BASE_URL + "apofaseis.asp"

    # Headers copied from browser analysis, User-Agent is crucial
    headers = {
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'accept-language': 'en-US,en;q=0.9,el;q=0.8', # Added Greek language preference
        'cache-control': 'max-age=0',
        'content-type': 'application/x-www-form-urlencoded',
        'origin': 'https://areiospagos.gr',
        'referer': initial_form_page_url, # Referer should be the form page
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36' # Example User-Agent
    }
    session.headers.update({'user-agent': headers['user-agent']}) # Set User-Agent for the session

    # Form data payload for the POST request
    payload = {
        'X_TMHMA': sector_code,    # The code for Civil/Criminal etc.
        'submit_krit': '',      # Name of the submit button seems to be this
        'X_SUB_TMHMA': '1',     # Seems to default to '1', might relate to sub-sector selection
        'X_TELESTIS_number': '1', # Operator for decision number (1=equals, not used here)
        'x_number': '',         # Decision number (empty means all)
        'X_TELESTIS_ETOS': '1', # Operator for year (1=equals)
        'x_ETOS': year,         # Target year
    }

    # --- Step 1: Establish Session (Optional but good practice) ---
    try:
        print(f"Making initial GET request to {initial_form_page_url} to establish session...")
        session.get(initial_form_page_url, headers={'user-agent': headers['user-agent']}, timeout=REQUEST_TIMEOUT)
        print("Session likely established. Cookies will now be managed by the session object.")
        # Introduce a small delay after initial contact
        time.sleep(REQUEST_DELAY * 2)
    except requests.exceptions.RequestException as e:
        print(f"Warning: Error during initial GET request: {e}. Proceeding, but session cookies might not be optimal.")

    all_decision_data_for_sector = []

    # --- Step 2: Perform Search POST Request ---
    try:
        print(f"Making POST request to {search_results_url}...")
        # print(f"Payload: {payload}") # Uncomment for debugging payload
        response = session.post(search_results_url, headers=headers, data=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        print(f"Search results page Status Code: {response.status_code}")

        # --- Step 3: Parse Search Results Page ---
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find category headers (often preceding the tables with links)
        # Selector targets <font> tags with specific color, containing '-' separating sector/subsector
        category_fonts = soup.select('font[color^="#000099"]')
        categories_text = [font.get_text(strip=True) for font in category_fonts if '-' in font.get_text()]
        print(f"Found {len(categories_text)} category headers: {categories_text}")

        # Find tables containing decision links
        decision_tables = soup.find_all('table')
        print(f"Found {len(decision_tables)} potential decision tables.")

        if len(categories_text) != len(decision_tables):
             print(f"Warning: Mismatch between category headers ({len(categories_text)}) and tables ({len(decision_tables)}). Parsing might be inaccurate.")
             # Attempt to proceed, assuming tables correspond sequentially to categories found

        table_idx = 0
        for cat_text in categories_text:
            if table_idx >= len(decision_tables):
                print(f"Warning: No more tables left for category '{cat_text}'. Stopping table processing.")
                break

            table = decision_tables[table_idx]
            table_idx += 1

            try:
                # Extract Sector and Subsector from the header text
                parts = cat_text.split('-', 1) # Split only on the first hyphen
                current_sector = parts[0].strip()
                current_subsector = parts[1].strip() if len(parts) > 1 else "Unknown"
                print(f"Processing Table for: Sector='{current_sector}', Subsector='{current_subsector}'")
            except IndexError:
                print(f"Warning: Could not parse category text '{cat_text}'. Using defaults.")
                current_sector = sector_name # Fallback to the main sector name
                current_subsector = "Unknown"

            # Find all decision links within the current table
            decision_links = table.select('a[href^="apofaseis_DISPLAY.asp"]')
            print(f"Found {len(decision_links)} decision links in this table.")

            # --- Step 4: Iterate Through Decision Links ---
            for link in decision_links:
                decision_text = link.get_text(strip=True)
                if "/" not in decision_text:
                    print(f"Warning: Skipping link with unexpected text: {decision_text}")
                    continue

                try:
                    decision_num = decision_text.split("/")[0].strip()
                    # Attempt to extract year from link text as fallback/confirmation
                    decision_year_text = decision_text.split("/")[1].strip()
                    if decision_year_text != year:
                         print(f"Warning: Year in link text ({decision_year_text}) differs from target year ({year}) for decision {decision_num}. Using target year.")
                except IndexError:
                    print(f"Warning: Could not parse decision number/year from link text: {decision_text}. Skipping link.")
                    continue

                # Initial data dictionary for this decision
                data = {
                    "decision_num": decision_num,
                    "year": year, # Use the target year consistently
                    "sector": current_sector,
                    "subsector": current_subsector,
                    "judges": None,
                    "court_intro_text": None,
                    "reasoning_text": None,
                    "decision_text": None,
                    "articles_ak": [], # Initialize as list (will be replaced by extract_decision_data)
                    "articles_kpolΔ": [],
                    "articles_pk": [],
                    "articles_kpd": [],
                }

                # Construct full URL for the decision page
                decision_url = BASE_URL + link.get('href')
                # print(f"Fetching decision {decision_num}/{year} from {decision_url}...") # Verbose logging

                # --- Step 5: Fetch and Parse Individual Decision Page ---
                time.sleep(REQUEST_DELAY) # Be polite!
                try:
                    decision_response = session.get(decision_url, timeout=REQUEST_TIMEOUT)
                    decision_response.raise_for_status()

                    # Parse the HTML of the decision page
                    decision_soup = BeautifulSoup(decision_response.content, 'html.parser')

                    # Extract detailed data using the helper function
                    extracted_data = extract_decision_data(decision_soup)

                    # Merge extracted data into the main data dictionary
                    data.update(extracted_data)

                    # Append the complete data for this decision to our list
                    all_decision_data_for_sector.append(data)
                    # print(f"Successfully processed decision {decision_num}/{year}.") # Verbose logging

                except requests.exceptions.Timeout:
                    print(f"Timeout Error fetching decision {decision_num}/{year} at {decision_url}. Skipping.")
                    continue
                except requests.exceptions.RequestException as e:
                    print(f"Error fetching or parsing decision {decision_num}/{year}: {e}. Skipping.")
                    continue
                except Exception as e:
                     print(f"Unexpected error processing decision {decision_num}/{year}: {e}. Skipping.")
                     continue

    except requests.exceptions.Timeout:
        print(f"Timeout Error during POST request for {sector_name} ({year}). Aborting crawl for this sector/year.")
    except requests.exceptions.RequestException as e:
        print(f"Fatal Error during POST request for {sector_name} ({year}): {e}. Aborting crawl for this sector/year.")
    except Exception as e:
        print(f"An unexpected error occurred during the crawl for {sector_name} ({year}): {e}")

    print(f"--- Finished crawl for Sector: {sector_name} ({sector_code}), Year: {year}. Found {len(all_decision_data_for_sector)} decisions. ---")
    return all_decision_data_for_sector

print("Main scraping function 'crawl_for_sector' defined.")
# -

# ### 3.3 Execute Scraping Process
#
# Now, we loop through the configured `TARGET_SECTORS` and `TARGET_YEAR` to run the scraper. The results are collected and then saved to a CSV file.
#
# **Note:** This cell can take a significant amount of time to run, depending on the number of decisions and the `REQUEST_DELAY`.

# +
all_scraped_data = []
print(f"Starting scraping process for year {TARGET_YEAR}...")

for sector_code, sector_name in TARGET_SECTORS.items():
    sector_data = crawl_for_sector(sector_code, sector_name, TARGET_YEAR)
    all_scraped_data.extend(sector_data)
    print(f"Collected {len(sector_data)} decisions for {sector_name} ({TARGET_YEAR}). Total collected: {len(all_scraped_data)}")

print("\n--- Scraping Complete ---")

if all_scraped_data:
    # Convert the list of dictionaries to a Pandas DataFrame
    df_scraped = pd.DataFrame(all_scraped_data)
    print(f"Successfully created DataFrame with {df_scraped.shape[0]} rows and {df_scraped.shape[1]} columns.")

    # Save the DataFrame to CSV
    try:
        df_scraped.to_csv(OUTPUT_CSV_FILENAME, index=False, encoding='utf-8')
        print(f"Data successfully saved to '{OUTPUT_CSV_FILENAME}'")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")

    # Display first few rows of the scraped data
    print("\nFirst 5 rows of scraped data:")
    print(df_scraped.head())
else:
    print("No data was scraped. The DataFrame and CSV file were not created.")
# -

# ## 4. Data Loading and Initial Inspection
#
# We load the data from the CSV file created in the previous step (or from a pre-existing file if the scraping step was skipped). We then inspect its basic properties like shape, columns, and data types.

# +
# --- Load Data ---
try:
    df = pd.read_csv(OUTPUT_CSV_FILENAME)
    print(f"Successfully loaded data from {OUTPUT_CSV_FILENAME}")
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumns:", df.columns.tolist())
    print("\nData Types:\n", df.dtypes)

except FileNotFoundError:
    print(f"Error: File not found at {OUTPUT_CSV_FILENAME}")
    print("Please ensure the scraping step ran successfully or the file exists.")
    # Optionally exit or handle this case differently depending on notebook flow
    # For now, we'll let subsequent cells fail if df is not loaded.
    df = None # Set df to None to indicate failure
except Exception as e:
    print(f"An error occurred while loading the CSV: {e}")
    df = None # Set df to None

# Basic check if DataFrame loaded
if df is None:
     print("\nHalting execution as DataFrame failed to load.")
     # In a notebook, you might just stop here or raise an error explicitly
     # raise RuntimeError("DataFrame could not be loaded.")
# -

# ## 5. Data Preparation
#
# Before analysis, we need to prepare the data. This includes:
# 1.  Converting the 'year' column to a numeric type.
# 2.  Defining a helper function to safely parse the string representations of article lists (e.g., "['361', '362']") into actual Python lists.

# +
# --- Helper Function for Parsing String Lists ---
def parse_string_list(list_str):
    """Safely parses a string that looks like a list into a Python list of strings."""
    if pd.isna(list_str) or not isinstance(list_str, str) or list_str.strip() == '[]' or list_str.strip() == '':
        return []
    try:
        # Use ast.literal_eval for safe evaluation of Python literals
        parsed_list = ast.literal_eval(list_str)
        if isinstance(parsed_list, list):
            # Ensure all elements are strings and strip whitespace
            return [str(item).strip() for item in parsed_list if str(item).strip()] # Also filter out empty strings after strip
        else:
            # print(f"Warning: Parsed literal is not a list: {list_str}")
            return [] # Return empty list if it wasn't actually a list
    except (ValueError, SyntaxError, TypeError) as e:
        # Handle cases where the string is malformed or not a list literal
        # print(f"Warning: Could not parse list string: '{list_str}'. Error: {e}")
        return []

print("Helper function 'parse_string_list' defined.")

# --- Data Type Conversion and Cleaning ---
if df is not None:
    print("\nPreparing DataFrame...")
    # Ensure 'year' is numeric
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        initial_rows = len(df)
        df.dropna(subset=['year'], inplace=True) # Remove rows where year couldn't be parsed
        rows_dropped = initial_rows - len(df)
        if rows_dropped > 0:
            print(f"Dropped {rows_dropped} rows with missing/invalid 'year' values.")
        df['year'] = df['year'].astype(int)
        print("'year' column converted to integer.")
    else:
        print("Warning: 'year' column not found.")

    # Optional: Fill NA for text columns if needed later (or handle during specific analysis)
    # text_cols = ['judges', 'court_intro_text', 'reasoning_text', 'decision_text']
    # for col in text_cols:
    #     if col in df.columns:
    #         df[col].fillna('', inplace=True)
    # print("Filled NA in text columns with empty strings.")

    print("\nData preparation complete. Current DataFrame info:")
    print(f"Shape: {df.shape}")
    print("Data Types:\n", df.dtypes.head()) # Show first few dtypes

else:
    print("Skipping data preparation as DataFrame is not loaded.")

# -

# ## 6. Analysis and Visualization
#
# Now we proceed with analyzing the prepared data and generating plots to understand the trends and characteristics of the legal decisions.

# ### 6.1 Plot 1: Decision Volume Over Time
#
# This plot shows the total number of decisions recorded for each year in the dataset.

# +
if df is not None and 'year' in df.columns and not df.empty:
    print("\n--- Generating Plot 1: Decision Volume Over Time ---")
    plt.figure(figsize=FIG_SIZE)
    sns.countplot(data=df, x='year', palette='viridis')
    plt.title('Number of Decisions per Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Decisions')
    # Improve x-axis readability if many years
    if df['year'].nunique() > 15:
         plt.xticks(rotation=45, ha='right', fontsize=10)
         # Optional: Show only every Nth label if very crowded
         # ax = plt.gca()
         # tick_labels = ax.get_xticklabels()
         # for i, label in enumerate(tick_labels):
         #     if i % 3 != 0: # Show every 3rd label
         #          label.set_visible(False)
    else:
         plt.xticks(df['year'].unique(), rotation=45, ha='right') # Ensure all unique years are shown if few

    plt.tight_layout()
    plt.show()
else:
    print("Skipping Plot 1: DataFrame not loaded or 'year' column missing/empty.")
# -

# ### 6.2 Plot 2: Distribution by Sector and Subsector
#
# These plots show the breakdown of decisions by the main legal sector (e.g., Civil, Criminal) and the more specific subsectors.

# +
if df is not None:
    print("\n--- Generating Plot 2a: Distribution by Sector ---")
    if 'sector' in df.columns and df['sector'].nunique() > 1:
        plt.figure(figsize=(8, 8))
        sector_counts = df['sector'].value_counts()
        plt.pie(sector_counts, labels=sector_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
        plt.title('Distribution of Decisions by Sector')
        plt.tight_layout()
        plt.show()
    elif 'sector' in df.columns:
         print("Skipping Sector pie chart (only one unique sector found or column missing).")
         print(f"Sectors found: {df['sector'].unique().tolist()}")
    else:
         print("Skipping Sector pie chart ('sector' column missing).")


    print("\n--- Generating Plot 2b: Distribution by Subsector (Top N) ---")
    if 'subsector' in df.columns and df['subsector'].nunique() > 0:
        plt.figure(figsize=FIG_SIZE)
        # Determine Top N based on available unique values, capped at a reasonable number (e.g., 20)
        top_n_subsectors = min(df['subsector'].nunique(), 20)
        subsector_counts = df['subsector'].value_counts().iloc[:top_n_subsectors]
        sns.barplot(y=subsector_counts.index, x=subsector_counts.values, palette='magma', orient='h')
        plt.title(f'Top {top_n_subsectors} Subsectors by Decision Count')
        plt.xlabel('Number of Decisions')
        plt.ylabel('Subsector')
        plt.tight_layout()
        plt.show()
    else:
        print("Skipping Subsector plot (column 'subsector' missing or has no data).")

else:
    print("Skipping Plot 2: DataFrame not loaded.")

# -

# ### 6.3 Plot 3: Trends in Sector Activity Over Time
#
# This line plot shows how the volume of decisions for each major sector has changed over the years present in the dataset. This requires data spanning multiple years to be meaningful.

# +
if df is not None and 'sector' in df.columns and 'year' in df.columns:
    print("\n--- Generating Plot 3: Trends in Sector Activity Over Time ---")
    if df['sector'].nunique() > 1 and df['year'].nunique() > 1:
        plt.figure(figsize=FIG_SIZE)
        # Group by year and sector, count decisions, handle potential missing combinations
        sector_yearly = df.groupby(['year', 'sector']).size().unstack(fill_value=0).stack().reset_index(name='count')

        sns.lineplot(data=sector_yearly, x='year', y='count', hue='sector', marker='o', palette='tab10')
        plt.title('Decision Volume by Sector Over Time')
        plt.xlabel('Year')
        plt.ylabel('Number of Decisions')
        plt.legend(title='Sector', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(df['year'].unique()) # Ensure all years are shown
        plt.grid(True, axis='y', linestyle='--')
        plt.tight_layout()
        plt.show()
    else:
         print("Skipping Sector Trends plot (requires multiple sectors AND multiple years in the data).")
         print(f"Unique sectors: {df['sector'].nunique()}, Unique years: {df['year'].nunique()}")
else:
     print("Skipping Plot 3: DataFrame not loaded or 'sector'/'year' columns missing.")
# -

# ### 6.4 Plot 4: Most Frequently Cited Articles
#
# These plots identify the most commonly cited articles for each legal code (AK, KPolD, PK, KPD) found in the dataset.

# +
if df is not None:
    print("\n--- Generating Plot 4: Most Frequently Cited Articles ---")

    # Use ARTICLE_COLUMNS defined in configuration
    article_columns_map = ARTICLE_COLUMNS

    found_any_articles = False
    for code_name, col_name in article_columns_map.items():
        if col_name in df.columns:
            print(f"\nAnalyzing articles for Code: {code_name} (Column: {col_name})")

            # Parse lists and flatten using the helper function
            # Apply the function to the series, then explode and filter NAs/empty lists
            parsed_series = df[col_name].apply(parse_string_list)
            all_articles = parsed_series.explode().dropna().tolist()

            if not all_articles:
                print(f"No articles found or parsed for {code_name} in column '{col_name}'.")
                continue

            found_any_articles = True
            # Count frequencies
            article_counts = Counter(all_articles)
            most_common = article_counts.most_common(TOP_N_ARTICLES)

            if not most_common:
                 print(f"No common articles to plot for {code_name}.")
                 continue

            # Prepare for plotting
            df_counts = pd.DataFrame(most_common, columns=['Article', 'Frequency'])

            # Plot
            plt.figure(figsize=FIG_SIZE)
            sns.barplot(data=df_counts, x='Frequency', y='Article', palette='coolwarm', orient='h')
            plt.title(f'Top {len(df_counts)} Most Cited Articles in {code_name}')
            plt.xlabel('Frequency (Number of Decisions Citing)')
            plt.ylabel('Article Number / Specification')
            plt.tight_layout()
            plt.show()
        else:
            print(f"Column '{col_name}' for code '{code_name}' not found in the dataset.")

    if not found_any_articles:
        print("\nNo article columns specified in ARTICLE_COLUMNS were found in the DataFrame.")

else:
    print("Skipping Plot 4: DataFrame not loaded.")

# -

# ### 6.5 Plot 5: Evolution of Specific Article Citations Over Time
#
# This plot demonstrates how to track the citation frequency of a *specific* article over the years. This is most insightful when analyzing key articles and requires data across multiple years.
#
# **Note:** We choose an example article (`'361'` from `AK`) here. You might want to change `key_article_code` and `key_article_num` to analyze a different article of interest.

# +
if df is not None:
    print("\n--- Generating Plot 5: Evolution of Key Article Citations Over Time ---")
    print("NOTE: This plot requires data across multiple years for meaningful trends.")

    # --- Configuration for this plot ---
    key_article_code = 'AK'     # Which code to look in (e.g., 'AK', 'KPolΔ')
    key_article_num = '361'   # The specific article number (as a string) to track
    # --- End Configuration ---

    key_article_col = ARTICLE_COLUMNS.get(key_article_code)

    if key_article_col and key_article_col in df.columns and 'year' in df.columns and df['year'].nunique() > 1:
        print(f"Tracking Article '{key_article_num}' from Code '{key_article_code}' (Column: '{key_article_col}')")

        # Create a boolean series: True if the key article is in the parsed list for that row
        # We need to re-apply the parsing here specific to this task
        def check_article_presence(list_str, article_to_find):
             parsed = parse_string_list(list_str)
             # Check for exact match or match at the beginning (e.g., "361" should match "361 παρ. 1")
             return any(item == article_to_find or item.startswith(article_to_find + " ") for item in parsed)

        df['cites_key_article'] = df[key_article_col].apply(lambda x: check_article_presence(x, key_article_num))

        # Group by year and sum the boolean column (True=1, False=0)
        yearly_key_counts = df.groupby('year')['cites_key_article'].sum().reset_index(name='Frequency')
        yearly_key_counts.sort_values('Year', inplace=True) # Ensure chronological order (already sorted by groupby usually)

        if not yearly_key_counts.empty and yearly_key_counts['Frequency'].sum() > 0:
            plt.figure(figsize=FIG_SIZE)
            sns.lineplot(data=yearly_key_counts, x='Year', y='Frequency', marker='o', color='green')
            plt.title(f'Frequency of Citation for Article {key_article_num} ({key_article_code}) Over Time')
            plt.xlabel('Year')
            plt.ylabel('Number of Decisions Citing Article')
            plt.xticks(yearly_key_counts['Year'].unique()) # Ensure all relevant years are shown
            plt.grid(True, axis='y', linestyle='--')
            plt.ylim(bottom=0) # Ensure y-axis starts at 0
            plt.tight_layout()
            plt.show()
        else:
             print(f"Could not generate trend for Article {key_article_num} ({key_article_code}). No citations found or data insufficient.")

        # Clean up the temporary column
        df.drop(columns=['cites_key_article'], inplace=True, errors='ignore')

    elif not (key_article_col and key_article_col in df.columns):
         print(f"Skipping Article Trend plot: Column '{key_article_col}' for code '{key_article_code}' not found.")
    elif 'year' not in df.columns:
         print("Skipping Article Trend plot: 'year' column missing.")
    elif df['year'].nunique() <= 1:
         print("Skipping Article Trend plot: Insufficient year data (only 1 year found).")
    else:
         print("Skipping Article Trend plot due to missing data or configuration.")

else:
     print("Skipping Plot 5: DataFrame not loaded.")

# -

# ### 6.6 Plot 6: Complexity of Decisions
#
# We can estimate the "complexity" of a decision by counting the number of unique articles cited across all relevant legal codes.
#
# *   **Plot 6a:** Shows the distribution of this complexity metric (how many decisions cite 1 article, 2 articles, etc.).
# *   **Plot 6b:** Shows how the *average* complexity per decision has changed over time (requires multiple years).

# +
if df is not None:
    print("\n--- Generating Plot 6: Complexity of Decisions (based on number of unique cited articles) ---")

    # Identify article columns present in the DataFrame based on ARTICLE_COLUMNS config
    article_cols_present = [col for col in ARTICLE_COLUMNS.values() if col in df.columns]

    if article_cols_present:
        print(f"Calculating complexity based on columns: {article_cols_present}")

        # Define function to count unique articles per row across specified columns
        def count_unique_articles_in_row(row):
            unique_articles = set()
            for col in article_cols_present:
                # Use the parsing helper function
                articles_list = parse_string_list(row[col])
                unique_articles.update(articles_list) # Add list items to set (handles duplicates automatically)
            return len(unique_articles)

        # Apply the function row-wise to create the complexity column
        df['num_articles_cited'] = df.apply(count_unique_articles_in_row, axis=1)
        print("Calculated 'num_articles_cited' for each decision.")

        # --- 6a: Histogram of Complexity Distribution ---
        plt.figure(figsize=FIG_SIZE)
        max_articles = df['num_articles_cited'].max()
        # Choose number of bins, ensuring at least 1 bin. Avoid too many bins if max_articles is large.
        num_bins = min(max(1, int(max_articles)), 30)
        sns.histplot(df['num_articles_cited'], bins=num_bins, kde=True, color='skyblue')
        plt.title('Distribution of Number of Unique Articles Cited per Decision')
        plt.xlabel('Number of Unique Articles Cited')
        plt.ylabel('Number of Decisions')
        plt.tight_layout()
        plt.show()

        # --- 6b: Average Complexity Over Time ---
        if 'year' in df.columns and df['year'].nunique() > 1:
            avg_complexity_yearly = df.groupby('year')['num_articles_cited'].mean().reset_index()
            plt.figure(figsize=FIG_SIZE)
            sns.lineplot(data=avg_complexity_yearly, x='year', y='num_articles_cited', marker='o', color='red')
            plt.title('Average Number of Unique Articles Cited per Decision Over Time')
            plt.xlabel('Year')
            plt.ylabel('Average Number of Unique Articles Cited')
            plt.xticks(avg_complexity_yearly['year'].unique()) # Show all years with data
            plt.grid(True, axis='y', linestyle='--')
            plt.ylim(bottom=0) # Ensure y-axis starts at 0
            plt.tight_layout()
            plt.show()
        elif 'year' in df.columns:
            print("Skipping Average Complexity Trend plot (only 1 year of data found).")
        else:
             print("Skipping Average Complexity Trend plot ('year' column missing).")

        # Optional: Clean up the created column if not needed later
        # df.drop(columns=['num_articles_cited'], inplace=True, errors='ignore')

    else:
        print("Skipping Complexity plots: No relevant article columns found in the DataFrame based on configuration.")
        print(f"(Looked for: {list(ARTICLE_COLUMNS.values())})")

else:
    print("Skipping Plot 6: DataFrame not loaded.")

# -

# ## 7. Conclusion
#
# This notebook demonstrated the process of scraping legal decision data from the Areios Pagos website and performing exploratory data analysis. Key insights include trends in decision volume, distribution across sectors, identification of frequently cited articles, and analysis of decision complexity over time. The generated plots provide a visual summary of these findings. Further analysis could involve natural language processing (NLP) on the text fields or more detailed statistical modeling.

# +
print("\n--- Analysis Notebook Complete ---")
# -
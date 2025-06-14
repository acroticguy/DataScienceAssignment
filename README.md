# Areios Pagos (Greek Supreme Court) Decision Analysis

This repository contains a Python project for scraping, analyzing, and visualizing legal decision data from the official website of the Areios Pagos, the Supreme Court of Greece.

The project is implemented as a Jupyter Notebook (`areios_pagos_analysis.ipynb`) and its corresponding Python script (`areios_pagos_analysis.py`). It performs two main tasks:
1.  **Data Extraction:** It scrapes legal decision metadata (year, sector, judges, etc.) and a list of cited articles from key legal codes (AK, KPolD, PK, KPD).
2.  **Data Analysis & Visualization:** It processes the scraped data and generates a series of plots to uncover trends, distributions, and patterns in the court's decisions.

## Features

- **Web Scraper:** A robust scraper built with `requests` and `BeautifulSoup` to navigate the Areios Pagos website and handle its search forms.
- **Data Parsing:** Uses regular expressions (Regex) to extract structured information like cited articles from unstructured text.
- **Configurable:** Easily change the target year and legal sectors for scraping by modifying variables in the configuration section.
- **Data Persistence:** Saves all scraped data into a clean, reusable `output_data.csv` file.
- **Exploratory Data Analysis (EDA):** A comprehensive analysis of the dataset, including:
  -   Decision volume over time.
  -   Distribution of decisions by legal sector and sub-sector.
  -   Frequency analysis of the most cited articles.
  -   Trends in citation frequency for specific articles.
  -   Analysis of decision "complexity" based on the number of unique articles cited.

## Repository Structure

```
├── .gitignore                 # Specifies files to be ignored by Git (e.g., generated data files).
├── README.md                  # This documentation file.
├── areios_pagos_analysis.ipynb # The main Jupyter Notebook for analysis and visualization.
├── areios_pagos_analysis.py   # A Python script version of the notebook, generated via Jupytext.
└── output_data.csv            # (Generated File) Scraped data is saved here by default.
```

## Prerequisites

- Python 3.7+
- The required Python libraries, which can be installed via pip.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Install the required packages:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
    Install the dependencies from a `requirements.txt` file. You can create this file with the following content:
    
    **requirements.txt:**
    ```
    requests
    beautifulsoup4
    pandas
    matplotlib
    seaborn
    notebook
    ```

    Then, run the installation command:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

The analysis can be run using either the Jupyter Notebook (recommended for interactive use and viewing plots) or the Python script.

### Method 1: Using the Jupyter Notebook (Recommended)

1.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

2.  **Open the notebook:**
    In the browser window that opens, navigate to and open `areios_pagos_analysis.ipynb`.

3.  **Run the cells:**
    You can run the cells sequentially by pressing `Shift + Enter`.
    -   **Configuration:** Before running, you can adjust the parameters in the **Configuration** section (cell #2), such as `TARGET_YEAR` or `TARGET_SECTORS`.
    -   **Scraping:** The **Data Extraction** section will begin the scraping process. This may take a significant amount of time depending on the number of decisions and the `REQUEST_DELAY` setting. The output will be saved to `output_data.csv`.
    -   **Analysis:** Once the CSV file is created, the subsequent cells will load the data and generate all the analytical plots directly in the notebook. If you already have an `output_data.csv` file, you can skip the scraping cells and run the analysis part directly.

### Method 2: Using the Python Script

You can execute the entire workflow from the command line.

1.  **Configure the script (Optional):**
    Open `areios_pagos_analysis.py` in a text editor and modify the variables in the **Configuration** section as needed.

2.  **Run the script:**
    ```bash
    python areios_pagos_analysis.py
    ```
    The script will execute all steps: scraping, saving the data to `output_data.csv`, and then generating and displaying the plots one by one.

## Disclaimer

- Web scraping can be fragile. The scraper in this project is designed for the website structure as of its creation date. **If the Areios Pagos website is updated, the scraper may break.**
- Please scrape responsibly. The script includes a `REQUEST_DELAY` to avoid overwhelming the server. Do not set this value too low.

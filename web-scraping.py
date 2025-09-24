import requests
from bs4 import BeautifulSoup
import os
import time

BASE_URL = "https://ghalii.org"
START_URL = "https://ghalii.org/judgments/GHASC/"

# make a folder for cases
os.makedirs("cases", exist_ok=True)

def scrape_case_list(page_url):
    response = requests.get(page_url)
    soup = BeautifulSoup(response.text, "html.parser")

    # each case is in <div class="view-content"> → <h3><a>
    for case in soup.select("tr > td > div > a"):
        case_title = case.text.strip()
        case_url = BASE_URL + case["href"]

        print(f"🔎 Found case: {case_title}")
        scrape_case_page(case_title, case_url)

def scrape_case_page(case_title, case_url):
    response = requests.get(case_url)
    soup = BeautifulSoup(response.text, "html.parser")

    # look for a PDF download link
    pdf_link = soup.find("a", string="Download PDF")
    if pdf_link:
        pdf_url = BASE_URL + pdf_link["href"]
        download_pdf(case_title, pdf_url)
    else:
        print(f"⚠️ No PDF found for {case_title}")

def download_pdf(case_title, pdf_url):
    response = requests.get(pdf_url)
    safe_title = case_title.replace(" ", "_").replace("/", "-")  # avoid bad filenames
    filepath = os.path.join("cases", f"{safe_title}.pdf")

    with open(filepath, "wb") as f:
        f.write(response.content)
    print(f"✅ Downloaded: {filepath}")

# run the scraper
scrape_case_list(START_URL)
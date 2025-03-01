import requests
from bs4 import BeautifulSoup

# Constants
SEARCH_QUERY = "security engineering jobs nearby Oakland"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15"
GOOGLE_URL = "https://www.google.com/search?q="
OUTPUT_FILE = "job_descriptions.txt"

# Function to scrape job descriptions from Google search results
def scrape_job_descriptions(query):
    headers = {"User-Agent": USER_AGENT}
    response = requests.get(GOOGLE_URL + query, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    # Debugging: Print the entire HTML
    with open("debug_html.html", "w", encoding="utf-8") as debug_file:
        debug_file.write(soup.prettify())
    print("Debug HTML saved to debug_html.html.")

    # Update this selector based on the actual HTML structure
    job_descriptions = []
    for result in soup.find_all("div", class_="BNeawe s3v9rd AP7Wnd"):  # Update this line
        job_descriptions.append(result.get_text())

    return job_descriptions

# Function to write job descriptions to a text file
def write_to_text_file(job_descriptions, filename):
    with open(filename, "w", encoding="utf-8") as file:
        for idx, description in enumerate(job_descriptions):
            file.write(f"Job {idx + 1}:\n")
            file.write(description + "\n")
            file.write("\n" + "=" * 80 + "\n")  # Separator between jobs
    print(f"Job descriptions written to {filename}.")

# Main function
def main():
    # Scrape job descriptions
    job_descriptions = scrape_job_descriptions(SEARCH_QUERY)
    print(f"Scraped {len(job_descriptions)} job descriptions.")

    # Write job descriptions to a text file
    write_to_text_file(job_descriptions, OUTPUT_FILE)

if __name__ == "__main__":
    main()

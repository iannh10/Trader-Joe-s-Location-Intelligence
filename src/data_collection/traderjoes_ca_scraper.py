import csv
import os
import random
import re
import time
import requests
from bs4 import BeautifulSoup


ca_url = "https://locations.traderjoes.com/ca/"
base_url = "https://locations.traderjoes.com"
out_path = "data/trader_joes/tj_locations_raw.csv"

headers = {
    "User-Agent": "Mozilla/5.0"
}


def wait_a_little():
    # pause between requests so we dont get blocked
    seconds = random.uniform(2, 4)
    time.sleep(seconds)


def get_page(url):
    try:
        response = requests.get(url, headers=headers, timeout=20)
        wait_a_little()

        if response.status_code != 200:
            print("bad response:", response.status_code, "for", url)
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        return soup
    except Exception as e:
        print("error getting page:", url)
        print(e)
        return None


def clean_text(text):
    if text is None:
        return ""
    # get rid of line breaks and extra whitespace
    text = text.replace("\n", " ")
    # collapse multiple spaces into one
    parts = text.split()
    text = " ".join(parts)
    return text


def get_phone_number(soup, all_text):
    phone = ""

    # first try to find a clickable phone link
    all_links = soup.find_all("a", href=True)
    for link in all_links:
        href = link["href"]
        if href.startswith("tel:"):
            phone = clean_text(link.get_text())
            # sometimes the text is empty but the href has the number
            if phone == "":
                phone = href.replace("tel:", "")
                phone = phone.strip()
            break

    # if we still dont have a phone number, try regex on the page text
    if phone == "":
        phone_match = re.search(r"\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}", all_text)
        if phone_match is not None:
            phone = phone_match.group(0)

    return phone


def get_address_parts(lines):
    street = ""
    city = ""
    state = ""
    zip_code = ""

    # try to find a line that starts with a number (thats usually the street)
    # then the next line should have city, state zip
    found_it = False
    for i in range(len(lines)):
        current_line = lines[i]

        # check if line starts with a digit
        has_number = re.match(r"^\d+\s+", current_line)
        if has_number:
            street = current_line

            # look at the line right after
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                match = re.search(r"^(.*?),\s*([A-Z]{2})\s+(\d{5})", next_line)
                if match is not None:
                    city = match.group(1)
                    state = match.group(2)
                    zip_code = match.group(3)
                    found_it = True
                    break

    # backup plan: just search every line for the city/state/zip pattern
    if not found_it:
        for i in range(len(lines)):
            line = lines[i]
            match = re.search(r"^(.*?),\s*([A-Z]{2})\s+(\d{5})", line)
            if match is not None:
                city = match.group(1)
                state = match.group(2)
                zip_code = match.group(3)
                break

    return street, city, state, zip_code


def get_store_info(store_link):
    soup = get_page(store_link)
    if soup is None:
        return None

    store_info = {}
    store_info["store_url"] = store_link

    # get the store name from the h1 tag
    title_tag = soup.find("h1")
    if title_tag is not None:
        store_info["store_name"] = clean_text(title_tag.get_text())
    else:
        store_info["store_name"] = ""

    # grab all the text on the page for searching
    all_text = soup.get_text()
    store_info["phone"] = get_phone_number(soup, all_text)

    # split the page text into non empty lines
    raw_lines = all_text.split("\n")
    lines = []
    for line in raw_lines:
        cleaned = clean_text(line)
        if cleaned != "":
            lines.append(cleaned)

    street, city, state, zip_code = get_address_parts(lines)

    store_info["street"] = street
    store_info["city"] = city
    store_info["state"] = state
    store_info["zip_code"] = zip_code

    return store_info


def get_city_links():
    print("getting city links from main CA page")
    soup = get_page(ca_url)
    if soup is None:
        print("could not load the main page!")
        return []

    # use a set to avoid duplicates
    city_links = set()

    all_links = soup.find_all("a", href=True)
    for a in all_links:
        href = a["href"]
        # city pages look like /ca/some-city/
        if href.startswith("/ca/") and href != "/ca/":
            full_url = base_url + href
            city_links.add(full_url)

    # turn set into list
    city_links = list(city_links)
    print("found", len(city_links), "cities")
    return city_links


def get_store_links(city_links):
    store_links = set()
    city_num = 1

    for i in range(len(city_links)):
        city_url = city_links[i]
        print("city", city_num, "/", len(city_links), city_url)
        soup = get_page(city_url)

        if soup is not None:
            all_links = soup.find_all("a", href=True)
            for a in all_links:
                href = a["href"]
                # store pages end with a number
                if href.startswith("/ca/") and re.search(r"\d+/?$", href):
                    full_url = base_url + href
                    store_links.add(full_url)

        city_num = city_num + 1

    store_links = list(store_links)
    print("found total stores:", len(store_links))
    return store_links


def save_to_csv(rows, file_path):
    columns = ["store_name", "street", "city", "state", "zip_code", "phone", "store_url"]

    f = open(file_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()

    count = 0
    for row in rows:
        writer.writerow(row)
        count = count + 1

    f.close()
    print("wrote", count, "rows to", file_path)


if __name__ == "__main__":
    if not os.path.exists("data/trader_joes"):
        os.makedirs("data/trader_joes")

    city_links = get_city_links()
    store_links = get_store_links(city_links)

    all_data = []
    store_num = 1

    for i in range(len(store_links)):
        store_url = store_links[i]
        print("scraping store", store_num, "/", len(store_links), store_url)
        info = get_store_info(store_url)
        if info is not None:
            all_data.append(info)
        else:
            print("  skipped, could not get info")
        store_num = store_num + 1

    save_to_csv(all_data, out_path)
    print("saved to", out_path)
    print("total stores scraped:", len(all_data))

# svd_downloader.py
# copied from https://github.com/rijulg/svd-downloader

import json
import logging
import re
import sys
from argparse import ArgumentParser
from logging import Logger
from pathlib import Path

import dask
import requests
from bs4 import BeautifulSoup
from dask.diagnostics.progress import ProgressBar


class Downloader:
    base_url = "https://stimmdb.coli.uni-saarland.de"
    max_speaker_id = 2742
    session = requests.Session()

    def __init__(self, out_path: str, refetch_links: bool, logger: Logger):
        self.logger = logger
        self.out_path = out_path
        self.refetch_links = refetch_links
        self.links_file = f"{out_path}/data.json"

    def run(self):
        self.logger.info("Running downloader")
        self.logger.info(f"Saving at: {self.out_path}")
        if not Path(self.links_file).is_file() or self.refetch_links:
            self.logger.info("Saving file links")
            self.save_all_file_links()
        with open(self.links_file, "r") as links_file:
            data = json.load(links_file)
        self.download_data(data)

    def download_data(self, data):
        jobs = []
        self.logger.info("Setting up download jobs")
        for key, row in data.items():
            if row is not None:
                gender = row["gender"]
                for session in row["sessions"]:
                    session_id = session["session_id"]
                    classification = session["classification"]
                    for file in session["files"]:
                        job = self.download_file(
                            key=key,
                            session_id=session_id,
                            gender=gender,
                            classification=classification,
                            file=file,
                        )
                        jobs += [job]
        self.logger.info("Downloading files")
        with ProgressBar():
            dask.compute(*jobs)

    @dask.delayed
    def download_file(self, key, session_id, gender, classification, file):
        data_path = f"{self.out_path}/{classification}/{gender}/{key}/{session_id}"
        Path(data_path).mkdir(parents=True, exist_ok=True)
        file_id = file.split("=")[1]
        file_path = f"{data_path}/{file_id}.wav"
        doc = requests.get(file)
        with open(file_path, "wb") as f:
            f.write(doc.content)

    def save_all_file_links(self):
        session = self.db_session()
        ids = list(range(self.max_speaker_id))
        pages = [f"{self.base_url}/details.php4?SprecherID={number}" for number in ids]
        jobs = [self.extract_links_from_page(session, page) for page in pages]
        with ProgressBar():
            data = dict(zip(ids, dask.compute(*jobs)))
        Path(self.links_file).parent.mkdir(exist_ok=True, parents=True)
        with open(self.links_file, "w") as link_file:
            json.dump(data, link_file, indent=4)

    @dask.delayed
    def extract_links_from_page(self, session, url):
        response = session.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        gender = self.get_gender(soup.find("div", class_="title").text)
        valid_classifications = ["healthy", "pathological"]
        data = {"page": url, "gender": gender, "sessions": []}
        for sess in soup.findAll("table", class_="sessiondetails"):
            session_id = self.find_session_id(sess)
            classification = self.identify_classification(sess)
            recording_date, age = self.find_dates(sess)
            pathologies = self.find_pathologies(sess)
            sess_data = {
                "session_id": session_id,
                "classification": classification,
                "age": age,
                "recording_date": recording_date,
                "pathologies": pathologies,
            }
            if classification in valid_classifications:
                sess_data["files"] = self.get_file_links(sess)
            else:
                print(f"Invalid classification(={classification}) found at url: {url}")
            data["sessions"] += [sess_data]
        return data

    def find_session_id(self, sess):
        row = sess.find("tr", class_="titleactive")
        cell = row.find("td")
        link = cell.find("a")
        return link["name"]

    def find_dates(self, sess):
        rows = sess.find_all("tr", class_="detailsactive")
        for row in rows:
            cells = row.find_all("td")
            if "date of recording" in cells[0].text.lower():
                recording_date = cells[1].find("span").text
                age = re.findall(r"\((\d*)\)", cells[1].text)[0]
                return recording_date, age

    def find_pathologies(self, sess):
        rows = sess.find_all("tr", class_="detailsactive")
        for row in rows:
            cells = row.find_all("td")
            if "pathologies" in cells[0].text.lower():
                return cells[1].text.strip()

    def db_session(self):
        session = requests.Session()
        session.post(f"{self.base_url}/index.php4", data={"sb_lang": "English"})
        session.post(
            f"{self.base_url}/index.php4",
            data={"sb_search": "Database request", "sb_sent": "Accept"},
        )
        return session

    def identify_classification(self, sess):
        row = sess.find("tr", class_="detailsactive")
        classification = row.find_all("td")[1].text
        return classification

    def get_file_links(self, sess):
        files = sess.find_all("a", attrs={"target": "PLAY"})
        file_links = [f"{self.base_url}/{x.get('href')}" for x in files]
        return file_links

    def get_gender(self, title: str):
        title = title.lower()
        if "female" in title:
            return "female"
        if "male" in title:
            return "male"
        return "unknown"


def get_logger():
    log_handler = logging.StreamHandler(sys.stdout)
    log_formatter = logging.Formatter(fmt="[%(asctime)s] %(levelname)-8s:: %(message)s")
    log_handler.setFormatter(log_formatter)

    logger = logging.Logger("SVD Downloader")
    logger.addHandler(log_handler)
    return logger


if __name__ == "__main__":
    parser = ArgumentParser("SVD Downloader")
    parser.add_argument(
        "out_path", type=str, default="SVD", help="The location of where to store data"
    )
    parser.add_argument(
        "--refetch-links",
        action="store_true",
        help="Whether to refresh the list of links",
    )
    args = parser.parse_args()

    logger = get_logger()
    downloader = Downloader(
        out_path=args.out_path, refetch_links=args.refetch_links, logger=logger
    )
    downloader.run()

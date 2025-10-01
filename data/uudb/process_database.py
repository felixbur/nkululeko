#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified UUDB XML -> per-emotion CSV exporter
Outputs one CSV per emotion with columns: file,label
file: absolute path C:\\Users\\Atoz_\\.nkululeko\\data\\UUDB\\UUDB\\var\\<session>\\<wav>
label: weak|strong (mean of annotators, floor)
"""
from __future__ import annotations
import os
import csv
import glob
import math
import xml.etree.ElementTree as ET
from statistics import mean
from typing import List

ROOT = f"./UUDB"
SESSIONS_DIR = os.path.join(ROOT, "Sessions")
VAR_DIR = os.path.join(ROOT, "var")
OUT_DIR = os.path.join(ROOT, "datasets")
EMOTIONS = ["Pleasantness", "Arousal", "Dominance", "Credibility", "Interest", "Positivity"]
NS = {"u": "http://uudb.speech-lab.org/ns/0.9"}

os.makedirs(OUT_DIR, exist_ok=True)


def parse_rating_to_label(values: List[float]) -> str:
    if not values:
        return None
    v = int(math.floor(mean(values)))
    return "weak" if v <= 4 else "strong"


def find_wav_from_comments(utt_elem: ET.Element) -> str:
    ucomment = utt_elem.find('u:UtteranceComment', NS)
    if ucomment is None:
        return None
    for comment in ucomment.findall('u:Comment', NS):
        cs = comment.attrib.get('CommentStrings', '')
        if cs and cs.lower().endswith('.wav'):
            return cs
    return None


def process_sessions():
    # accumulator: emotion -> list of (file,label) for train and test
    rows_train = {e: [] for e in EMOTIONS}
    rows_test = {e: [] for e in EMOTIONS}
    if not os.path.isdir(SESSIONS_DIR):
        print(f"Sessions dir not found: {SESSIONS_DIR}")
        return rows_train, rows_test

    # prepare ordered list of session subdirectories
    session_dirs = [d for d in sorted(os.listdir(SESSIONS_DIR)) if os.path.isdir(os.path.join(SESSIONS_DIR, d))]
    n = len(session_dirs)
    if n == 0:
        return rows_train, rows_test

    # split index: take first 70% (7:3), sequential order, not random
    split_idx = int(n * 0.7)
    if split_idx < 1 and n >= 1:
        split_idx = 1
    if split_idx >= n and n > 1:
        split_idx = n - 1

    train_sessions = set(session_dirs[:split_idx])
    test_sessions = set(session_dirs[split_idx:])

    for session in session_dirs:
        session_path = os.path.join(SESSIONS_DIR, session)
        if not os.path.isdir(session_path):
            continue
        xmls = glob.glob(os.path.join(session_path, "*.xml"))
        if not xmls:
            continue
        # pick the xml that matches session name if exists
        xml_path = xmls[0]
        for x in xmls:
            if os.path.basename(x).lower() == f"{session.lower()}.xml":
                xml_path = x
                break
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            print(f"Failed to parse {xml_path}: {e}")
            continue

        for utt in root.findall('u:Utterance', NS):
            utt_id = utt.attrib.get('UtteranceID', '').strip()
            channel = utt.attrib.get('Channel', '').strip()
            if not utt_id or not channel:
                continue
            recording = f"{session}{channel}_{utt_id}"  # e.g. C001R_001

            wav_name = find_wav_from_comments(utt)
            if not wav_name:
                # fallback to recording.wav
                wav_name = f"{recording}.wav"

            # ensure path uses VAR_DIR/<prefix>/<wav_name>
            prefix = wav_name[:4]
            pref_path = os.path.join(VAR_DIR, prefix, wav_name)

            # search recursively for any matching filename under VAR_DIR
            try:
                matches = glob.glob(os.path.join(VAR_DIR, '**', wav_name), recursive=True)
            except Exception:
                matches = []

            wav_path = None
            # 1) if pref_path exists use it
            if os.path.exists(pref_path):
                wav_path = os.path.abspath(pref_path)
            else:
                # 2) prefer any match whose parent dir equals prefix
                for m in matches:
                    if os.path.basename(os.path.dirname(m)).lower() == prefix.lower():
                        wav_path = os.path.abspath(m)
                        break
                # 3) else prefer the first match if any
                if wav_path is None and matches:
                    wav_path = os.path.abspath(matches[0])
                # 4) else default to pref_path (include subdir even if missing)
                if wav_path is None:
                    wav_path = os.path.abspath(pref_path)

            est = utt.find('u:EmotionalState', NS)
            if est is None:
                continue

            # determine whether this session is train or test
            target = 'train' if session in train_sessions else 'test'

            # collect ratings per emotion
            for emo in EMOTIONS:
                vals = []
                for r in est.findall('u:Rating', NS):
                    v = r.attrib.get(emo)
                    if v is None:
                        continue
                    try:
                        vals.append(float(v))
                    except Exception:
                        continue
                label = parse_rating_to_label(vals)
                if label is None:
                    continue
                if target == 'train':
                    rows_train[emo].append((wav_path, label))
                else:
                    rows_test[emo].append((wav_path, label))

    return rows_train, rows_test


def write_csvs(rows_train, rows_test):
    for emo in EMOTIONS:
        data_train = rows_train.get(emo, [])
        data_test = rows_test.get(emo, [])

        outp_train = os.path.join(OUT_DIR, f"dataset_{emo}_train.csv")
        with open(outp_train, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(['file', 'label'])
            for f, l in data_train:
                writer.writerow([f, l])

        outp_test = os.path.join(OUT_DIR, f"dataset_{emo}_test.csv")
        with open(outp_test, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(['file', 'label'])
            for f, l in data_test:
                writer.writerow([f, l])

        print(f"Wrote {len(data_train)} train / {len(data_test)} test rows -> {outp_train}, {outp_test}")


if __name__ == '__main__':
    rows_train, rows_test = process_sessions()
    write_csvs(rows_train, rows_test)
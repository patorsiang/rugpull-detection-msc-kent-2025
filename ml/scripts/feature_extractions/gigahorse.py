import subprocess
import os
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import sys
sys.path.append(str(Path.cwd().parents[1]))
from scripts.utils import get_grouping_opcode_sequence

def assess_hex_file(host_path, address, filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    container_path = "/tmp"
    container_hex = f"{container_path}/hex/{filename}"

    try:
        # 1. Create a temporary container
        print("[*] Creating container...")
        container_id = subprocess.check_output([
            "docker", "create", "-v", f"{host_path}:{container_path}",
            "gigahorse", container_hex
        ]).decode().strip()

        print("[*] Starting analysis...")
        subprocess.run(["docker", "start", "-a", container_id], check=True)

        output_name = f"{address}.result.json"
        output_path = os.path.join(output_dir, output_name)

        print("[*] Copying results.json...")
        subprocess.run([
            "docker", "cp",
            f"{container_id}:/opt/RugPull/gigahorse/results.json",
            output_path
        ], check=True)

        # Copy all *.facts files from the container to the hex directory
        hex_dir = os.path.join(output_dir, address)

        os.makedirs(hex_dir, exist_ok=True)
        print("[*] Copying all .facts files...")
        subprocess.run([
            "docker", "cp",
            f"{container_id}:/opt/RugPull/gigahorse/.temp/{address}/.",  # note the dot (.) at the end
            hex_dir
        ], check=True)

        print(f"[+] results.json saved to: {output_path}")

        with lock:
            print(f"[✓] {address} completed")
        print(f"[✓] {address} completed")
    except subprocess.CalledProcessError:
        print("[✗] Error during analysis or copy.")
        with lock:
            print(f"[✗] {address} failed")
        print(f"[✗] {address} failed")
    finally:
        subprocess.run(["docker", "rm", container_id], stdout=subprocess.DEVNULL)
    return 0

def get_n_gram_from_gigahorse_assessment(host_path, addresses, files):
    # assess each file in the set
    output_dir = os.path.join(host_path, "out")
    output_list = os.listdir(output_dir)

    for address, path in zip(addresses, files):
        if address not in output_list:
            assess_hex_file(host_path, address, path, output_dir)

    # extract n_gram from out folder
    opcode_seq = []

    for directory in output_list:
        if '.' not in directory:
            address = directory.lower()
            opcode_path = os.path.join(output_dir, f"{directory}/Statement_Opcode.facts")
            op_df = pd.read_csv(opcode_path, sep="\t", header=None, names=["stmt", "opcode"])

            opcodes = op_df["opcode"].tolist()

            freq = dict()
            freq['opcode_sequence'] = get_grouping_opcode_sequence(opcodes)
            freq['address'] = address
            opcode_seq.append(freq)

    seq_df = pd.DataFrame(opcode_seq).fillna(0).set_index('address')

    vectorizer = CountVectorizer(ngram_range=(2, 3), analyzer='word', max_features=1000)
    X_ngrams = vectorizer.fit_transform(seq_df['opcode_sequence'])

    feature_df = pd.DataFrame(X_ngrams.toarray(), columns=vectorizer.get_feature_names_out())

    return feature_df

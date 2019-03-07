# Trending Words and Topics Extraction
This project will extract trending words and topics from the provided CSV file. Please refer to `documentation.pdf` for various details.


## Installation
Run the following command in sequence to generate `topic.txt` which will capture the trending words and topics

    virtualenv -p python3 venv/
    source venv/bin/activate
    pip install -r requirements.txt
    python3 trending_words.py

## Other Info
1. If you want to change the parameters, please use `config/config.yaml`;
2. Simple logging is provided in `info.log` file.


#!/usr/bin/env python
"""
Download a NLM terminology resource
Code modified from https://github.com/danizen/nlm-demo-notebook

"""
import os
import argparse
import requests
from tqdm import tqdm
from lxml import html as lhtml

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--apikey", type=str, default=None, required=True)
parser.add_argument("--url", type=str, default=None, required=True)
args = parser.parse_args()

# Step 1 - obtain a Ticket Granting Ticket (TGT)
session = requests.session()
response = session.post(
    'https://utslogin.nlm.nih.gov/cas/v1/api-key',
    data={'apikey': args.apikey}
)

# extract the TGT
doc = lhtml.fromstring(response.text)
print(doc)
TGT = doc.xpath("//form/@action")[0]

# Step 2 - obtain a Service Ticket (ST)
r = session.post(TGT, data={'service': args.url})
ST = r.text

print(f"Ticket Granting Ticket (TGT): {TGT}")
print(f"Service Ticket (ST):          {ST}")
print(f"URL: {args.url}")

# Step 3 - Download release file
r = session.get(f'{args.url}?ticket={ST}', stream = True)
total_size = int(r.headers.get('content-length', 0))
print(f'Total size {total_size/1e+9:2.2f} GB')

with open(os.path.basename(args.url), 'wb') as outfile:
    with tqdm(total=total_size / (512 * 1024.0), unit='B',
              unit_scale=True,
              unit_divisor=1024) as progress_bar:
            for chunk in r.iter_content(chunk_size=512 * 1024):
                if chunk: # filter out keep-alive new chunks
                    outfile.write(chunk)
                    progress_bar.update(len(chunk))
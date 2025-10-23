import pandas as pd
from rapidfuzz import fuzz
import re

# Files
names_file = 'meeting_attendees.xlsx'
emails_file = 'all_emails.xlsx'
output_file = 'meeting_attendees_aligned_v2.xlsx'

# Read
names_df = pd.read_excel(names_file)
emails_df = pd.read_excel(emails_file)

name_col = names_df.columns[0]
email_col = emails_df.columns[1] if emails_df.shape[1] >= 2 else emails_df.columns[0]

emails_list = emails_df[email_col].dropna().astype(str).tolist()

# Normalizers
split_re = re.compile(r'[\._\-\d]+')

def normalize(s):
    s = re.sub(r"\(.*?\)", "", str(s))
    s = re.sub(r"\[.*?\]", "", s)
    s = s.replace(',', ' ')
    s = s.replace("'", '')
    s = s.strip().lower()
    s = re.sub(r'\s+', ' ', s)
    return s


def email_key_tokens(email):
    local = email.split('@')[0].lower()
    return [t for t in re.split(split_re, local) if t]

# Matching
results = []
uncertain = []

for idx, row in names_df.iterrows():
    name = row[name_col]
    norm = normalize(name)
    best_score = 0
    best_email = ''
    second_best = None
    for e in emails_list:
        # score using token overlap and fuzzy
        tokens = email_key_tokens(e)
        token_score = 0
        for t in tokens:
            if t and t in norm:
                token_score += 25
        fuzz_score = fuzz.token_sort_ratio(norm, e.split('@')[0])
        score = max(fuzz_score, token_score)
        if score > best_score:
            second_best = (best_email, best_score)
            best_score = score
            best_email = e
    # Normalize to 0-1
    conf = round(best_score / 100.0, 2)
    if conf >= 0.6:
        results.append((name, best_email, conf))
    elif best_email:
        uncertain.append((name, best_email, conf))
    else:
        uncertain.append((name, '', 0.0))

# Build DataFrame
out = pd.DataFrame(results, columns=['Name', 'Matched Email', 'Confidence'])
unc_df = pd.DataFrame(uncertain, columns=['Name', 'Top Candidate', 'Confidence'])

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    names_df.to_excel(writer, index=False, sheet_name='Original')
    out.to_excel(writer, index=False, sheet_name='Matches')
    unc_df.to_excel(writer, index=False, sheet_name='Uncertain')

print(f'Wrote {output_file} — matches: {len(out)}, uncertain: {len(unc_df)}')

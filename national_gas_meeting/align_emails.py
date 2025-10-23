import re
import pandas as pd
from collections import defaultdict

# File paths
input_file = 'meeting_attendees.xlsx'
output_file = 'meeting_attendees_aligned.xlsx'

# Emails provided by user
emails = [
    'stuart.elliott@qemsolutions.com',
    'larissa.sidarto@metyis.com',
    'david.hart@newtoneurope.com',
    'chris.williams2@macegroup.com',
    'palzor.lama@emerson.com',
    'adrian.webb@fishergerman.co.uk',
    'mohammad.hassaan@cngservices.co.uk',
    'alex@alexbarnesassociates.co.uk',
    'andy.giles@holtenergyadvisors.com',
    'chris.beech@worley.com',
    'chris.burn@pxlimited.com',
    'christopher.mccartney@gmo-ni.com',
    'cornelia.waymouth@gmsl.co.uk',
    'jessica.cotton@engie.com',
    'knapperd@tcd.ie',
    'dave.whiting@interconnector.com',
    'david.goodall@stevevick.com',
    'ndayal@assystem.com',
    'debbie@eua.org.uk',
    'dharmesh.jadavji@baringa.com',
    'joanne.farrar@atkinsrealis.com',
    'fatima.sow@elexon.co.uk',
    'garry.mcloughlin@gmo-ni.com',
    'julian.green@aspentech.com',
    'ashley.haigh@chubbfs.com',
    'ray.hicks@dnv.com',
    'julie.cox@energy-uk.org.uk',
    'joerg_keil@siemens-energy.com',
    'kirsty@eci-partners.co.uk',
    'lauren.rowe@futurebiogas.com',
    'mark.hewett@bfygroup.co.uk',
    'matthew.atkinson@sefe.eu',
    'mdaws@summit-evolution.com',
    'melvyn.wilson@troocost.com',
    'mike@flagshipenergy.co.uk',
    'paul.johnstone@swagelokcentral.co.uk',
    "paul.oconnor@gmsl.co.uk",
    'pete.hughes@gmsl.co.uk',
    'jack.piper@energysecurity.gov.uk',
    'jack.piper@energysecurity.gov.uk',
    'rhiannon.mcevoy@gmo-ni.com',
    'rob.seaton@energysecurity.gov.uk',
    'stephen.ohare@gmo-ni.com',
    'steven.puetz@simanalytica.com',
    'thomas.mccartney@gmo-ni.com',
    'timc@ceramics-uk.org',
    'yahya.amodo@cemex.com'
]

# Normalization helpers

split_re = re.compile(r'[\._\-\d]+')

def normalize_name(name):
    # Remove parentheses and their contents, and extra descriptors like (External)
    name = re.sub(r"\(.*?\)", "", name)
    # Remove brackets
    name = re.sub(r"\[.*?\]", "", name)
    # Remove extra commas and roles after commas like 'Ball, Andrew (M4)'
    name = name.replace(',', ' ')
    name = name.replace("'", '')
    name = name.strip()
    name = re.sub(r'\s+', ' ', name)
    return name.lower()

def name_tokens(name):
    name = normalize_name(name)
    tokens = [t for t in re.split(r"[\s,]+", name) if t]
    # remove common words
    tokens = [t for t in tokens if t not in ('unverified','external','generation','and','tra','gb')]
    return tokens

def email_local_tokens(email):
    local = email.split('@')[0].lower()
    # split on dots, underscores, dashes and digits
    tokens = [t for t in re.split(split_re, local) if t]
    return tokens

# Read input excel
try:
    df = pd.read_excel(input_file)
except Exception as e:
    print(f"Failed to read {input_file}: {e}")
    raise

# Determine which column contains names (first column)
name_col = df.columns[0]

# Prepare tokens for names
name_token_map = {}
for idx, val in df[name_col].fillna('').items():
    name_token_map[idx] = name_tokens(str(val))

# Matching
email_assignments = defaultdict(list)  # idx -> list of matched emails
unmatched_emails = set(emails)
possible_matches = []

for email in emails:
    e_tokens = email_local_tokens(email)
    best_score = 0
    best_idx = None
    tie = False

    for idx, n_tokens in name_token_map.items():
        if not n_tokens:
            continue
        # count how many name tokens appear in email tokens or vice versa
        match_count = sum(1 for t in n_tokens if any(t == et or t.startswith(et) or et.startswith(t) for et in e_tokens))
        # also consider initials (first initial + last name)
        initials_match = 0
        if len(n_tokens) >= 2:
            first, last = n_tokens[0], n_tokens[-1]
            # email like jsmith or j.smith
            if any(et == f'{first[0]}{last}' or et == f'{first[0]}_{last}' or et == f'{first[0]}.{last}' for et in e_tokens):
                initials_match = 1
        score = (match_count + initials_match) / max(1, len(n_tokens))

        if score > best_score:
            best_score = score
            best_idx = idx
            tie = False
        elif score == best_score and score > 0:
            tie = True

    # Accept matches with score >= 0.6 and not tie
    if best_idx is not None and best_score >= 0.6 and not tie:
        email_assignments[best_idx].append((email, best_score))
        unmatched_emails.discard(email)
    else:
        # keep as unassigned for now
        pass

# Build result dataframe
df['Matched Email'] = ''
df['Match Confidence'] = ''
for idx, matches in email_assignments.items():
    # if multiple, pick highest score
    matches_sorted = sorted(matches, key=lambda x: x[1], reverse=True)
    df.at[idx, 'Matched Email'] = matches_sorted[0][0]
    df.at[idx, 'Match Confidence'] = round(matches_sorted[0][1], 2)

# Collect emails that might match multiple people or had low confidence
low_confidence = []
for email in list(unmatched_emails):
    e_tokens = email_local_tokens(email)
    candidates = []
    for idx, n_tokens in name_token_map.items():
        if not n_tokens:
            continue
        match_count = sum(1 for t in n_tokens if any(t == et or t.startswith(et) or et.startswith(t) for et in e_tokens))
        initials_match = 0
        if len(n_tokens) >= 2:
            first, last = n_tokens[0], n_tokens[-1]
            if any(et == f'{first[0]}{last}' or et == f'{first[0]}.{last}' or et == f'{first[0]}_{last}' for et in e_tokens):
                initials_match = 1
        score = (match_count + initials_match) / max(1, len(n_tokens))
        if score > 0:
            candidates.append((idx, score))
    if candidates:
        # sort and keep top 3
        candidates_sorted = sorted(candidates, key=lambda x: x[1], reverse=True)[:3]
        low_confidence.append((email, candidates_sorted))
        unmatched_emails.discard(email)

# Remaining unmatched_emails are totally unmatched
still_unmatched = set([e for e in emails if e not in [m for lst in email_assignments.values() for m,_ in lst]])
# Also include those we moved to low_confidence
for e, _ in low_confidence:
    if e in still_unmatched:
        still_unmatched.discard(e)

# Write output excel with two sheets
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df.to_excel(writer, index=False, sheet_name='Aligned')
    # Unmatched clear list
    unmatched_df = pd.DataFrame({'Uncertain Matches (email)': [e for e,_ in low_confidence],
                                 'Top Candidates (row idx and score)': [str(c) for _,c in low_confidence]})
    unmatched_df.to_excel(writer, index=False, sheet_name='Uncertain')
    still_unmatched_df = pd.DataFrame({'Unmatched Emails': list(still_unmatched)})
    still_unmatched_df.to_excel(writer, index=False, sheet_name='Unmatched')

# Summary print
matched_count = df['Matched Email'].astype(bool).sum()
print(f"Wrote {output_file}")
print(f"Total attendees: {len(df)}")
print(f"Direct matched emails: {matched_count}")
print(f"Low-confidence (uncertain) emails: {len(low_confidence)}")
print(f"Totally unmatched emails: {len(still_unmatched)}")

# Print details for quick review
print('\nSample matches:')
print(df.loc[df['Matched Email']!='', [name_col, 'Matched Email', 'Match Confidence']].head(20).to_string(index=False))

if low_confidence:
    print('\nLow-confidence email suggestions:')
    for email, cands in low_confidence:
        print(email, '->', cands)

if still_unmatched:
    print('\nStill unmatched:')
    for e in still_unmatched:
        print(e)

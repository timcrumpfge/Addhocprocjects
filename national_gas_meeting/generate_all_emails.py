import pandas as pd

input_file = 'meeting_attendees.xlsx'
output_file = 'all_emails.xlsx'

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
    'paul.oconnor@gmsl.co.uk',
    'pete.hughes@gmsl.co.uk',
    'jack.piper@energysecurity.gov.uk',
    'rhiannon.mcevoy@gmo-ni.com',
    'rob.seaton@energysecurity.gov.uk',
    'stephen.ohare@gmo-ni.com',
    'steven.puetz@simanalytica.com',
    'thomas.mccartney@gmo-ni.com',
    'timc@ceramics-uk.org',
    'yahya.amodo@cemex.com'
]

# Read the existing attendees file
try:
    df = pd.read_excel(input_file)
except Exception as e:
    print(f"Failed to read {input_file}: {e}")
    raise

# Determine existing emails (if any) in the second column
if df.shape[1] >= 2:
    email_col = df.columns[1]
    existing_emails = df[email_col].dropna().astype(str).tolist()
else:
    existing_emails = []

# Combine unique emails preserving order, append unmatched at the end
combined = existing_emails[:]
for e in emails:
    if e not in combined:
        combined.append(e)

# Build rows: pair original names where available, otherwise blank
names = df[df.columns[0]].astype(str).tolist()
rows = []
max_len = max(len(names), len(combined))
for i in range(max_len):
    name = names[i] if i < len(names) else ''
    email = combined[i] if i < len(combined) else ''
    rows.append({'Name': name, 'Email': email})

out_df = pd.DataFrame(rows)
out_df.to_excel(output_file, index=False)
print(f"Wrote {output_file} with {len(out_df)} rows (emails: {len(combined)})")

# Validate by reading back
test = pd.read_excel(output_file)
print('Validation read OK — sample:')
print(test.head(5))

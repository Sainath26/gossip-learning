"""
Imports
"""
import hashlib
import re
from pathlib import Path
import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSIncome



"""
Script Variables
"""
DATA = Path(__file__).resolve().parent / 'data'
# Official 2018 ACS PUMS Data Dictionary (U.S. Census Bureau), the source of the code groups.
# The checksum makes sure the mapping always comes from this exact file
REFERENCE = Path(__file__).resolve().parent / 'reference' / 'PUMS_Data_Dictionary_2018.txt'
REFERENCE_SHA256 = '843a18e15b6cb7c7326261d478947a1415c4674c62419a8d01946a5a5df4d5fd'
# Place-of-birth regions by POBP code range, as laid out in the dictionary
POBP_REGIONS = [(1, 56, 'US state'), (60, 99, 'US territory'), (100, 199, 'Europe'), (200, 299, 'Asia'),
                (300, 399, 'Americas'), (400, 499, 'Africa'), (500, 599, 'Oceania and other')]
# 50 states plus Puerto Rico (no DC): the 51 nodes, as in FairFed
STATES = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY',
          'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND',
          'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']
NUMERICAL = ['AGEP', 'WKHP']  # age and usual hours worked per week; the other features are categorical
PROTECTED_COLUMN = {'race': 'RAC1P', 'sex': 'SEX'}


"""
Helpers
"""
def _protected(df, attribute):
    # The protected attribute as 0/1: 1 for the privileged group (white, or male), 0 otherwise.
    # Race is white ('White alone', RAC1P = 1) vs non-white (all other codes), as in FairFed
    if attribute == 'race':
        return (df['RAC1P'] == 1).astype(int)
    if attribute == 'sex':
        return (df['SEX'] == 1).astype(int)
    raise ValueError(f'unknown protected attribute: {attribute}')

def _stratified_pick(df, fraction, attribute, rng):
    # True for a random `fraction` of the rows of every state x protected group x label stratum.
    # `fraction` is one number, or one value per row
    strata = [df['ST'], _protected(df, attribute), df['label']]
    keep = (fraction * df.groupby(strata)['ST'].transform('size')).round()
    # Random order within each stratum; pick the first `keep` rows of each
    order = pd.Series(rng.random(len(df)), index=df.index).groupby(strata).rank(method='first')
    return order <= keep

def _dictionary_codes(lines, variable):
    # {code: label} from one variable's section of the dictionary, e.g. 10: 'MGR-Chief Executives ...'
    header = re.compile(r'[A-Z][A-Z0-9]*\s+(Character|Numeric)\s+\d+')
    start = next(i for i, line in enumerate(lines) if header.match(line) and line.split()[0] == variable)
    codes = {}
    for line in lines[start + 1:]:
        if header.match(line):  # the next variable's section starts
            break
        m = re.match(r'\s+(\d+)\s+\.(.*)', line)
        if m:
            codes[int(m.group(1))] = m.group(2).strip()
    return codes


def _categories(df, attribute):
    # The sorted category list of every categorical input, from the full (grouped) table.
    # Built once, not per sample, so every seed and every node has the same input layout:
    # column k of the one-hot input always means the same thing
    inputs = [c for c in ACSIncome.features if c not in NUMERICAL and c != PROTECTED_COLUMN[attribute]]
    return {col: sorted(df[col].unique()) for col in inputs}

"""
Main Functions
"""
def load_acs(year=2018):
    # All 51 states as one table: the 10 ACSIncome features, the label (income above $50,000)
    # and the state. Built once from the ACS files (downloaded into data/acs on first use),
    # then read from a cache in data/processed
    cache = DATA / 'processed' / f'acsincome_{year}.npz'
    if cache.exists():
        with np.load(cache) as f:
            return pd.DataFrame({col: f[col] for col in f.files})

    source = ACSDataSource(survey_year=str(year), horizon='1-Year', survey='person',
                           root_dir=str(DATA / 'acs'))
    frames = []
    # One state at a time: the raw files have about 280 columns, so loading all at once uses a lot of RAM
    for st in STATES:
        features, label, _ = ACSIncome.df_to_pandas(source.get_data(states=[st], download=True))
        frames.append(features.assign(label=label['PINCP'].astype(int), ST=st))
    df = pd.concat(frames, ignore_index=True)
    # folktables returns codes as floats; they are whole numbers
    df[ACSIncome.features] = df[ACSIncome.features].astype(int)

    cache.parent.mkdir(parents=True, exist_ok=True)
    # State codes are saved as plain 2-letter strings, so the cache loads without pickle
    np.savez_compressed(cache, **{col: df[col].to_numpy(dtype='U2' if col == 'ST' else None)
                                  for col in df.columns})
    return df



def group_codes(df, reference=REFERENCE):
    # Replaces detailed occupation (OCCP, ~530 codes) and place-of-birth (POBP, ~280 codes)
    # with their major groups from the official dictionary
    raw = Path(reference).read_bytes()
    if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA256:
        raise ValueError(f'{reference} is not the expected 2018 PUMS Data Dictionary')
    lines = raw.decode('latin-1').splitlines()

    # Occupation: each label starts with its major group, e.g. '0010 .MGR-Chief Executives'
    labels = _dictionary_codes(lines, 'OCCP')
    occp = {code: label[:3] for code, label in labels.items() if re.match(r'[A-Z]{3}-', label)}
    # A few labels have no prefix: such a code joins the group whose code range contains it
    # (1555 -> ENG, 8000 -> PRD), or else forms its own group (9920, unemployed)
    ranges = {g: (min(c for c in occp if occp[c] == g), max(c for c in occp if occp[c] == g))
              for g in set(occp.values())}
    for code in labels.keys() - occp.keys():
        occp[code] = next((g for g, (lo, hi) in ranges.items() if lo <= code <= hi), f'OCCP{code}')

    # Place of birth: region by code range
    pobp = {code: next(name for lo, hi, name in POBP_REGIONS if lo <= code <= hi)
            for code in _dictionary_codes(lines, 'POBP')}

    # Every code in the data must be in the dictionary
    for col, mapping in (('OCCP', occp), ('POBP', pobp)):
        unknown = set(df[col]) - mapping.keys()
        if unknown:
            raise ValueError(f'{col} codes not in the dictionary: {sorted(unknown)[:10]}')
    return df.assign(OCCP=df['OCCP'].map(occp), POBP=df['POBP'].map(pobp))




def draw_sample(df, fraction, min_rows, seed, attribute):
    # Stratified sample: the same fraction of every state x protected group x label stratum.
    # A state whose share would fall below min_rows is sampled at a higher fraction (still
    # stratified), so it keeps about min_rows rows. Rows keep their index in the full table,
    # so a seed's sample can be saved as row numbers
    state_size = df.groupby('ST')['ST'].transform('size')
    state_fraction = np.maximum(fraction, np.minimum(min_rows, state_size) / state_size)
    return df[_stratified_pick(df, state_fraction, attribute, np.random.default_rng([seed, 0]))]


def split(sample, test_fraction, seed, attribute):
    # The shared test set: test_fraction of every state x protected group x label stratum of
    # the sample. Returns (train, test); each state's train rows are that node's data
    is_test = _stratified_pick(sample, test_fraction, attribute, np.random.default_rng([seed, 1]))
    return sample[~is_test], sample[is_test]






def encode(train, test, cats, attribute, standardise_on='train'):
    # Model inputs for both splits, following Fair2Zero: categorical features as integer codes
    # (expanded to one-hot only inside the model, to save memory), numerical features
    # standardised, and the protected attribute left out of the inputs.
    # standardise_on: 'train' uses the training rows' mean and spread, 'all' uses train and test
    stats = train if standardise_on == 'train' else pd.concat([train, test])
    mean, std = stats[NUMERICAL].mean(), stats[NUMERICAL].std()

    def arrays(part):
        codes = np.stack([pd.Categorical(part[col], categories=cats[col]).codes for col in cats], axis=1)
        if (codes < 0).any():
            raise ValueError('a category is missing from the category list')
        return {'cat': codes.astype(np.int16),
                'num': ((part[NUMERICAL] - mean) / std).to_numpy(np.float32),
                'y': part['label'].to_numpy(np.int8),
                'pa': _protected(part, attribute).to_numpy(np.int8),
                'state': part['ST'].to_numpy(dtype='U2')}

    return arrays(train), arrays(test)

def prepare(config, seed):
    # Everything a run needs for one seed: each state's training data, the shared test set and
    # the size of each categorical input (for the model's one-hot layer).
    # The seed's sample and split are saved as row numbers in data/splits/<settings>/, and reused
    # on later runs, so gossip and federated learning train on exactly the same rows
    data, attribute = config['data'], config['fairness']['protected_attribute']
    if config['fairness']['income_threshold'] != 50000:
        raise ValueError('ACSIncome labels income above $50,000; other thresholds are not supported')

    full = load_acs(data['survey_year'])
    if data['group_occupation_birthplace']:
        full = group_codes(full)

    # The folder name holds every setting that changes which rows are picked
    folder = DATA / 'splits' / (f"{data['survey_year']}_{attribute}_frac{data['sample_fraction']}"
                                f"_min{data['min_rows_per_state']}_test{data['test_fraction']}")
    path = folder / f'seed_{seed:03d}.npz'
    if path.exists():
        with np.load(path) as f:
            train, test = full.loc[f['train']], full.loc[f['test']]
    else:
        sample = draw_sample(full, data['sample_fraction'], data['min_rows_per_state'], seed, attribute)
        train, test = split(sample, data['test_fraction'], seed, attribute)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, train=train.index.to_numpy(), test=test.index.to_numpy())

    cats = _categories(full, attribute)
    train_arrays, test_arrays = encode(train, test, cats, attribute, data['standardise_on'])
    # One entry per node: that state's training rows
    nodes = {st: {k: v[train_arrays['state'] == st] for k, v in train_arrays.items() if k != 'state'}
             for st in STATES}
    return {'nodes': nodes, 'test': test_arrays, 'cat_sizes': [len(c) for c in cats.values()]}





if __name__ == '__main__':
    # Builds the cache and seed 1's split for the settings in configs.json, then prints a check
    import json
    config = json.loads((Path(__file__).resolve().parent / 'configs.json').read_text())
    out = prepare(config, seed=1)
    sizes = {st: len(n['y']) for st, n in out['nodes'].items()}
    print(f"{len(sizes)} nodes, {sum(sizes.values()):,} training rows, {len(out['test']['y']):,} test rows")
    print(f"smallest node {min(sizes, key=sizes.get)} ({min(sizes.values()):,}), "
          f"largest {max(sizes, key=sizes.get)} ({max(sizes.values()):,})")
    print(f"one-hot width {sum(out['cat_sizes']) + out['test']['num'].shape[1]}")
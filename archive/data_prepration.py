"""
This is a stand alone file for
data prepration and making splits for training and testing the model/

Stage 1 of the pipeline. It produces everything the training code reads:

  prepare_data()  data/adult.zip (UCI Adult)  ->  data/processed/adult.npz
                  Loads the train and test files together (48,842 rows), encodes
                  them into 106 numeric features, and saves X, y (income > 50K)
                  and pa (protected attribute: sex, 1 = Male, 0 = Female).

  make_splits()   data/processed/adult.npz    ->  data/splits/seed_XXX.npz
                  For every seed: holds out a stratified 20% test set, then
                  splits the remaining rows across 10 clients with different
                  sizes, sex ratios and positive rates (a non-IID partition).
                  Only row indices are stored, never copies of the data.

Run directly (python data_prepration.py) to do both steps. run.py calls the
same two functions when the files are missing.
"""

import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd # type: ignore


def _read_schema(z):
    """
    Parse adult.names: lines like "age: continuous." or "sex: Female, Male."
    The target column is not listed there, so it is appended as "income".
    
    Returns:
      columns     column names in file order (14 features + "income")
      continuous  the names of the numeric columns; all others are categorical
    """

    text = z.read("adult.names").decode("latin-1")
    columns, continuous = [], set()
    for line in text.splitlines():
        # A feature line is "<lower-case name>: <values>." The regex skips the
        # free-text description that makes up the rest of adult.names
        m = re.match(r"^([a-z][a-z-]*):\s*(.+)\.\s*$", line)
        if m:
            name, values = m.groups()
            columns.append(name)
            if values.strip() == "continuous":
                continuous.add(name)
    columns.append("income")
    return columns, continuous


def prepare_data(zip_path="data/adult.zip", out_path="data/processed/adult.npz"):
    """
    Takes in a zip file and prepare the data
    Steps: read both UCI files -> build labels y and protected attribute pa ->
    standardise numeric columns -> one-hot encode categorical columns ->
    check the result against the paper (48,842 x 106, PR ratio 2.78) -> save.
    Returns (X, y, pa) as numpy arrays:
    X   float32, shape (48842, 106), model inputs (sex is NOT among them)
    y   0/1 labels, 1 = income > 50K
    pa  0/1 protected attribute, 1 = Male, 0 = Female
    """
    
    with zipfile.ZipFile(zip_path) as z:
        columns, continuous = _read_schema(z)
        # Reads one CSV member of the zip straight from memory.
        # skipinitialspace removes the space after each comma (", Male")
        read = lambda name, skip: pd.read_csv(
            io.BytesIO(z.read(name)), header=None, names=columns,
            skipinitialspace=True, skiprows=skip)
        # adult.test starts with a junk line "|1x3 Cross validator"
        # The paper uses all 48,842 rows, so the two files are stacked into one
        df = pd.concat([read("adult.data", 0), read("adult.test", 1)],
                       ignore_index=True)

    # Test labels end with "." (">50K."), so strip it before mapping
    df["income"] = df["income"].str.rstrip(".")
    y = df["income"].map({">50K": 1, "<=50K": 0}).to_numpy()
    pa = df["sex"].map({"Male": 1, "Female": 0}).to_numpy()

    # PA is excluded from the model input; "?" is kept as its own category
    # (rows with missing values are NOT dropped; that is how the paper gets
    # both 48,842 rows and exactly 106 features)
    features = df.drop(columns=["sex", "income"])
    numeric = [c for c in features.columns if c in continuous]
    categorical = [c for c in features.columns if c not in continuous]

    # Standardised on the full dataset, as in the paper (population std)
    # i.e. every numeric column ends up with mean 0 and variance 1. Fitting on
    # all rows (test included) leaks a little test information; this is kept
    # deliberately to match the paper
    num = features[numeric].astype(float)
    num = (num - num.mean()) / num.std(ddof=0)
    # One 0/1 column per category level, named like "workclass=Private"
    cat = pd.get_dummies(features[categorical], prefix_sep="=", dtype=float)
    X_df = pd.concat([num, cat], axis=1)
    print(X_df.head())
    X = X_df.to_numpy(dtype=np.float32)

    # Sanity checks
    # Shape matches the paper: 6 numeric + 100 one-hot columns = 106 features
    assert X.shape == (48842, 106), X.shape
    assert not np.isnan(X).any()
    assert set(np.unique(y)) == {0, 1} and set(np.unique(pa)) == {0, 1}
    # Positive-rate ratio P(y=1 | male) / P(y=1 | female). The paper reports
    # 2.78, which confirms that PA = 1 means male
    pr_ratio = y[pa == 1].mean() / y[pa == 0].mean()
    assert abs(pr_ratio - 2.78) < 0.01, pr_ratio

    # Print how many rows fall into each of the 4 (PA, Y) groups
    print(f"X: {X.shape}, PR ratio: {pr_ratio:.2f}")
    for a in (0, 1):
        for t in (0, 1):
            print(f"  PA={a}, Y={t}: {((pa == a) & (y == t)).sum()}")

    # Feature names are saved too, so columns can be identified later
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y, pa=pa,
                        feature_names=np.array(X_df.columns))
    return X, y, pa


def _draw_demand(rng, n_pool, share_women, pos_rate, n_clients,
                 alpha_size, delta_pa, delta_fair):
    # Rows each client asks for from each (PA, Y) group.
    # Columns follow the group code pa * 2 + y: (0,0), (0,1), (1,0), (1,1)
    #
    # Arguments:
    #   n_pool       number of training rows available to split
    #   share_women  fraction of the training pool with PA = 0
    #   pos_rate     [P(y=1 | women), P(y=1 | men)] in the training pool
    #   alpha_size   Dirichlet concentration for client sizes (smaller = more
    #                unequal sizes; the paper uses 2.0)
    #   delta_pa     spread of each client's share of women around the global one
    #   delta_fair   spread of each client's positive rates around the global ones
    #
    # Returns an int array of shape (n_clients, 4). These are only requests;
    # make_splits() checks them against what is actually available.

    # Client sizes: random fractions that sum to 1 (Dirichlet), times the pool
    sizes = rng.dirichlet(np.full(n_clients, alpha_size)) * n_pool
    # PA-diversity: shifts the client's share of women
    # (additive Gaussian noise, clipped so every client keeps both sexes)
    women = np.clip(share_women + delta_pa * rng.standard_normal(n_clients),
                    0.05, 0.95)
    # Fairness-diversity: scales the positive rate of each PA group by a random
    # factor, so small rates (women ~11%) stay positive instead of hitting 0
    # rates[:, 0] is the client's positive rate for women, rates[:, 1] for men
    rates = np.clip(
        pos_rate * np.exp(delta_fair * rng.standard_normal((n_clients, 2))),
        0.01, 0.99)
    # Turn (share of women, positive rate per sex) into the fraction of the
    # client that comes from each of the 4 groups; each row sums to 1
    frac = np.column_stack([women * (1 - rates[:, 0]), women * rates[:, 0],
                            (1 - women) * (1 - rates[:, 1]),
                            (1 - women) * rates[:, 1]])
    return np.floor(sizes[:, None] * frac).astype(int)


def make_splits(processed_path="data/processed/adult.npz", out_dir="data/splits",
                seeds=range(1, 101), n_clients=10, test_frac=0.2,
                alpha_size=2.0, min_client_size=100,
                delta_pa=0.1, delta_fair=0.5, max_tries=1000):
# Per seed: a stratified test split, then the rest partitioned into clients.
# Only row indices into adult.npz are saved.
#
# Output files:
#   <out_dir>/config.json     the arguments used, for reproducibility
#   <out_dir>/seed_XXX.npz    test_idx, client_0 ... client_{n_clients-1}
#
# delta_pa and delta_fair are not given in the paper; these values were
# chosen so FedAvg's unfairness lands near the paper's numbers.
# max_tries bounds how often a partition is redrawn when a client ends up
# below min_client_size.
    data = np.load(processed_path)
    y, pa = data["y"], data["pa"]
    # Group code of every row: 0 = (F, <=50K), 1 = (F, >50K),
    #                          2 = (M, <=50K), 3 = (M, >50K)
    strata = pa * 2 + y

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = dict(processed_path=str(processed_path), seeds=list(seeds),
                  n_clients=n_clients, test_frac=test_frac,
                  alpha_size=alpha_size, min_client_size=min_client_size,
                  delta_pa=delta_pa, delta_fair=delta_fair)
    (out / "config.json").write_text(json.dumps(config, indent=2))

    for seed in seeds:
        # One generator per seed drives everything below, so each split can be
        # rebuilt exactly from its seed
        rng = np.random.default_rng(seed)

        # Test split: 20% of each (PA, Y) group
        # (shuffle each group's row indices and take the first 20%), so the
        # test set has the same sex/label mix as the whole dataset
        test_idx = np.concatenate([
            rng.permutation(np.flatnonzero(strata == g))
            [:round(test_frac * (strata == g).sum())] for g in range(4)])
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx)

        # Global proportions of the training pool
        y_tr, pa_tr = y[train_idx], pa[train_idx]
        share_women = (pa_tr == 0).mean()
        pos_rate = np.array([y_tr[pa_tr == 0].mean(), y_tr[pa_tr == 1].mean()])
        # Rows available in each of the 4 groups
        avail = np.bincount(strata[train_idx], minlength=4)

        # Draw client requests until every client has at least min_client_size
        # rows. The for/else raises only if the loop never hits "break"
        for _ in range(max_tries):
            demand = _draw_demand(rng, len(train_idx), share_women, pos_rate,
                                  n_clients, alpha_size, delta_pa, delta_fair)
            # If a group is over-requested, scale every client's request down
            # (this mostly happens for the small group of women earning >50K;
            # rows are never sampled twice, so requests must fit the supply)
            scale = np.minimum(1.0, avail / np.maximum(demand.sum(axis=0), 1))
            demand = np.floor(demand * scale).astype(int)
            if demand.sum(axis=1).min() >= min_client_size:
                break
        else:
            raise RuntimeError(f"seed {seed}: no valid partition found")

        # Hand out rows without replacement, so clients never share rows
        # For each group: shuffle its rows, then cut consecutive chunks, one
        # per client, of the requested lengths. Leftover rows go unused
        clients = [[] for _ in range(n_clients)]
        for g in range(4):
            pool = rng.permutation(train_idx[strata[train_idx] == g])
            ends = np.cumsum(demand[:, g])
            for k, chunk in enumerate(np.split(pool[:ends[-1]], ends[:-1])):
                clients[k].append(chunk)
        clients = [np.sort(np.concatenate(c)) for c in clients]

        # Sanity checks
        # No row may appear twice, either across clients or between a client
        # and the test set
        used = np.concatenate([test_idx, *clients])
        assert len(np.unique(used)) == len(used), "overlapping rows"
        assert min(len(c) for c in clients) >= min_client_size

        # Report how varied the clients are: size range, the range of their
        # male/female positive-rate ratios (global value is 2.78), and how
        # many training rows were left out
        ratios = [y[c][pa[c] == 1].mean() / max(y[c][pa[c] == 0].mean(), 1e-9)
                  for c in clients]
        print(f"seed {seed:3d}: sizes {min(map(len, clients))}-"
              f"{max(map(len, clients))}, client PR ratio "
              f"{min(ratios):.2f}-{max(ratios):.2f}, "
              f"unused {len(train_idx) - sum(map(len, clients))}")

        np.savez_compressed(out / f"seed_{seed:03d}.npz", test_idx=np.sort(test_idx),
                            **{f"client_{k}": c for k, c in enumerate(clients)})


if __name__ == "__main__":
    prepare_data()
    make_splits()

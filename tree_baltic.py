from pathlib import Path

import baltic as bt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from Bio import Phylo


# Input files
TREE_FILE = Path("foldmason_results/experimental_binding.nw")
META_FILE = Path("nipah_meta_withdenovo.csv")

FIXED_TREE_FILE = Path("experimental_binding_with_dummy_branch_lengths.nw")

# Output files
OUT_PNG = Path("experimental_binding_tree_denovo_id.png")
OUT_SVG = Path("experimental_binding_tree_denovo_id.svg")
OUT_PDF = Path("experimental_binding_tree_denovo_id.pdf")
OUT_NXS = Path("experimental_binding_tree_denovo_id.nxs")


COLOURS = {
    "yes": "#1b9e77",
    "no": "#d95f02",
    "unknown": "#7570b3",
}

LABELS = {
    "yes": "De novo: yes",
    "no": "De novo: no",
    "unknown": "De novo: unknown",
}


def clean_denovo(x):
    if pd.isna(x):
        return "unknown"

    x = str(x).strip().lower()

    if x in {"yes", "true", "1", "de novo"}:
        return "yes"

    if x in {"no", "false", "0", "not de novo"}:
        return "no"

    return "unknown"


def clean_structure_name(x):
    if pd.isna(x):
        return None

    x = str(x).strip()

    if x == "" or x.lower() in {"nan", "none"}:
        return None

    x = x.replace("\\", "/")
    x = x.split("/")[-1]
    x = x.replace(".cif", "")
    x = x.replace(".pdb", "")
    x = x.strip()

    if x == "":
        return None

    return x


def add_dummy_branch_lengths(input_tree, output_tree):
    tree = Phylo.read(input_tree, "newick")

    for clade in tree.find_clades():
        if clade.branch_length is None:
            clade.branch_length = 1.0

    Phylo.write(tree, output_tree, "newick")


def save_tree_as_nexus(input_tree, output_nexus):
    tree = Phylo.read(input_tree, "newick")
    Phylo.write(tree, output_nexus, "nexus")


def get_leaf_name(k):
    for attr in ["name", "numName"]:
        if hasattr(k, attr):
            value = getattr(k, attr)
            if value is not None:
                return str(value)

    return str(k)


def build_lookups(meta):
    required_cols = {
        "id",
        "de_novo",
        "esmfold_structure_prediction_file",
        "boltz2_structure_prediction_file",
    }

    missing = required_cols - set(meta.columns)

    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")

    meta["id_clean"] = meta["id"].fillna("Unknown ID").astype(str).str.strip()
    meta["de_novo_clean"] = meta["de_novo"].apply(clean_denovo)

    denovo_lookup = {}
    id_lookup = {}

    structure_cols = [
        "esmfold_structure_prediction_file",
        "boltz2_structure_prediction_file",
    ]

    for _, row in meta.iterrows():
        for col in structure_cols:
            structure_name = clean_structure_name(row[col])

            if structure_name is None:
                continue

            denovo_lookup[structure_name] = row["de_novo_clean"]
            id_lookup[structure_name] = row["id_clean"]

    return denovo_lookup, id_lookup, meta


def main():

    meta = pd.read_csv(META_FILE)

    denovo_lookup, id_lookup, meta = build_lookups(meta)

    add_dummy_branch_lengths(TREE_FILE, FIXED_TREE_FILE)

    save_tree_as_nexus(FIXED_TREE_FILE, OUT_NXS)

    tree = bt.loadNewick(str(FIXED_TREE_FILE))

    tree.traverse_tree()
    tree.sortBranches()

    fig, ax = plt.subplots(figsize=(10, 16))

    # Draw the tree
    tree.plotTree(
        ax,
        colour="#444444",
        linewidth=0.8,
        alpha=0.8,
    )

    missing_from_metadata = []

    for k in tree.Objects:
        if k.branchType == "leaf":
            leaf_name = get_leaf_name(k)

            denovo = denovo_lookup.get(leaf_name, "unknown")

            if leaf_name not in denovo_lookup:
                missing_from_metadata.append(leaf_name)

            ax.scatter(
                k.x,
                k.y,
                s=38,
                color=COLOURS[denovo],
                edgecolor="black",
                linewidth=0.25,
                zorder=10,
            )

    # Add ID labels
    SHOW_TIP_LABELS = True

    if SHOW_TIP_LABELS:
        xmax = max(k.x for k in tree.Objects)

        for k in tree.Objects:
            if k.branchType == "leaf":
                leaf_name = get_leaf_name(k)

                denovo = denovo_lookup.get(leaf_name, "unknown")
                display_label = id_lookup.get(leaf_name, leaf_name)

                ax.text(
                    xmax + 0.15,
                    k.y,
                    display_label,
                    va="center",
                    fontsize=5,
                    color=COLOURS[denovo],
                )

        ax.set_xlim(0, xmax + 4.5)

    handles = [
        mpatches.Patch(color=COLOURS[key], label=LABELS[key])
        for key in ["yes", "no", "unknown"]
    ]

    ax.legend(
        handles=handles,
        title="De novo status",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=False,
    )

    ax.set_title(
        "Experimental binding structures coloured by de novo status",
        fontsize=12,
    )

    ax.axis("off")
    plt.tight_layout()

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_SVG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")


    print("\nDe novo counts:")
    print(meta["de_novo_clean"].value_counts(dropna=False))

    if missing_from_metadata:
        print("\n not working")
        for name in missing_from_metadata:
            print(f"  {name}")
    else:
        print("\n ugh")


if __name__ == "__main__":
    main()
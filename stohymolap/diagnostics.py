import pandas as pd


def print_basic_statistics(
    discharge,
    precip,
    pet,
    peff
):

    print(
        pd.DataFrame({

            "Q": discharge[:10],
            "P": precip[:10],
            "PET": pet[:10],
            "Peff": peff[:10]

        })
    )

    print(
        f"\nQ mean={discharge.mean():.4f}"
    )

    print(
        f"P mean={precip.mean():.4f}"
    )

    print(
        f"PET mean={pet.mean():.4f}"
    )

    print(
        f"Peff mean={peff.mean():.4f}"
    )

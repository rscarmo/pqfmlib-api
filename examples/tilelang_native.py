"""Example: XYZ PQFM with TileLang and native connectivity on an NVIDIA GPU."""

from pqfmlib import XYZProjectiveQFM


if __name__ == "__main__":
    qfm = XYZProjectiveQFM(
        name_file="Toxicity_preprocessed_shuffled",
        seed=42,
        ideal=False,
        simulation=True,
        fakebackend=False,
        framework="tilelang",
        precision="double",  # Use "single" for FP32.
        expectation_method="exact",
        ibm_qpu="ibm_kingston",
        topology="native",
        q_enc=8,
        features_per_qubit=1,
        axes=("x", "y", "z"),
        encoding_mode="shared_feature",
        mps=False,
    )
    print(qfm.run())

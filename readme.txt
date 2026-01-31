Environment
- Python 3.8+
- PyTorch
- NumPy
- scikit-learn

--------------------
How to Run
--------------------
Each part of the assignment can be run with a SINGLE command,
as required by the assignment instructions.


Part 0: Bag-of-Words Models
--------------------------
Runs 2-layer and 3-layer BOW neural networks.

Command:
    python main.py --model BOW


Part 1: Deep Averaging Network (DAN)
-----------------------------------
- Part 1a: Pretrained GloVe 50d embeddings (fine-tuned)
- Part 1b: Randomly initialized 50d embeddings

Command:
    python main.py --model DAN


Part 2: Subword Deep Averaging Network (BPE)
--------------------------------------------
Runs Subword DAN models using Byte Pair Encoding (BPE) with
different vocabulary sizes.

Command:
    python main.py --model SUBWORDDAN

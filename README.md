# CS3602-Group-Project
The final project of CS3602(natural language processing---SJTU IEEE Honor Class).

## environment setup
    conda create -n slu python=3.6
    pip install torch==1.7.1

## file explanation
- `utils/args.py`: definitions of all related optional parameters
- `utils/initialization.py`: Initialize system settings, including setting random seed and GPU/CPU
- `utils/vocab.py`: Build a vocabulary for encoding input and output
- `utils/word2vec.py`: Load word vector
- `utils/example.py`: Read data
- `utils/batch.py`: Load input data as batched
- `model/slu_baseline_tagging.py`: Baseline model
- `scripts/slu_baseline.py`: Main program script

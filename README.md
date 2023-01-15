# CS3602-Group-Project
The final project of CS3602(natural language processing---SJTU IEEE Honor Class).

## Environment Setup
    conda create -n slu python=3.8
    conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

## File Explanation
- `SLUBase`: The pipeline for the baseline models
- `SLUBaseHistory`: The pipeline for the baseline models with the usage of history conversations
- `SLUBert`: The pipeline for our proposed model 'SLUBert' and 'SLUBert-MultiTurn' which leverages the history conversations
- `utils/args.py`: definitions of all related optional parameters
- `utils/initialization.py`: Initialize system settings, including setting random seed and GPU/CPU
- `utils/vocab.py`: Build a vocabulary for encoding input and output
- `utils/word2vec.py`: Load word vector
- `utils/example.py`: Read data
- `utils/batch.py`: Load input data as batches
- `model/slu_baseline_tagging.py`: Baseline model
- `scripts/slu_baseline.py`: Main program script

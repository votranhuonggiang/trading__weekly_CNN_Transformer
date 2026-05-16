# trading__weekly_CNN_Transformer

Weekly HOSE stock-selection research pipeline using QuestDB OHLCV data, engineered features, triple-barrier labels, and a CNN-Transformer training stack.

## Included for Colab

- `data/colab/`: GitHub-safe sharded training arrays
- `notebooks/colab_train_cnn_transformer.ipynb`: notebook to run on Google Colab
- `src/model.py`: CNN-Transformer classifier
- `src/train.py`: training loop and evaluation
- `src/colab_data.py`: loader for sharded arrays

## Colab workflow

1. Open Google Colab with GPU runtime enabled.
2. Upload or open `notebooks/colab_train_cnn_transformer.ipynb`.
3. Run the cells in order.

The notebook clones this repository, installs the requirements, loads the sharded arrays from `data/colab/`, and trains the model.

## Local preparation pipeline

Local data preparation requires access to QuestDB because raw market data is not bundled in the repo. The repo code includes:

- OHLCV ingestion
- daily feature engineering
- weekly triple-barrier labeling
- model dataset construction
- VNINDEX benchmark diagnostics

Main entrypoint:

```powershell
python run_project.py
```

![StockPatchTST_banner_proportional_1024x400](https://github.com/user-attachments/assets/89bad72a-5a85-4883-9666-171765832b15)
![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.6.0-%23ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

## 📚 Table of Contents
- [Input Features](#-input-features)
- [Key Features](#-key-features)
- [Target Selection Strategy](#-target-selection-strategy)
- [Model Hyperparameters](#-model-hyperparameters)
- [Labeling & Ranking](#%EF%B8%8F-labeling--ranking)
- [Return Evaluation](#-return-evaluation)
- [Case Study](#-case-study-specific-stocks)
- [Environment](#-environment)
- [Experiment Notebook](#-experiment-notebook)
- [License & Acknowledgements](#-license--acknowledgements)

## Overview of Deep Learning Model Evolution
A chronological overview of model improvements, from LSTM to Transformer variants.

![model_history](https://github.com/user-attachments/assets/3723d215-bd04-4fae-94a2-d0b3c8534d82)

## Transformer-Based Ranking Model for Stock Selection
📘 See [TRANSFORMER.md](docs/TRANSFORMER.md) to learn how the encoder works inside

![stock_patch_tst_model](https://github.com/user-attachments/assets/0d4c28c7-1bc2-4ea8-93cb-51896741d3c6)
> Although the feature dimension expands from 45 to 64, the input time series of 30 steps is reduced to 9 learned patches through Conv1D.
> This not only compresses temporal information but also abstracts each patch into a meaningful representation.
> The Transformer encoder then operates on these patch-level embeddings to learn inter-patch dependencies, rather than processing raw sequences.

---

## 🧠 Input Features
📘 See [FEATURES_TECHNICAL_INDICATORS.md](docs/FEATURES_TECHNICAL_INDICATORS.md) for the full list of features and indicator formulas.

The model was trained using a wide range of features including price movement, volatility, volume trends, market indices, and candlestick patterns. Below is the list of major features used for training.

> 💡 **Note on Feature Engineering**  
> All input features were transformed into **ratio-based values** to enhance generalizability across different price/volume levels.  
> Instead of applying sliding-window or historical normalization, features were expressed as relative changes (e.g., % gap from moving averages, rate-of-change, ratios).  
> Long-tail distributions (e.g., volume volatility ratios) were handled via simple log-scaling.  
> This approach was chosen to keep the feature space interpretable and stable across different stocks and timeframes.

### 🏭 Stock Metadata
- `industry_id`: Mid-level industry classification code
- `is_kospi`: Market type (KOSPI / KOSDAQ)

### 💹 Price Flow Indicators
- `close_rate`: Rate of change from previous close
- `open_to_close`, `high_to_low`: Intraday price movement ratios
- `rsi`: Relative Strength Index
- `macd_ratio`, `macd_signal_ratio`: MACD-related momentum ratios
- `macd_golden_cross`, `macd_dead_cross`: Momentum cross signals
- `atr_ratio`: Volatility ratio

### 📊 VWMA & Bollinger Bands
- `vwma5_gap`, `vwma20_gap`: VWMA (5/20-day) deviation from current close
- `vwma_bb_upper_ratio`, `vwma_bb_lower_ratio`, `vwma_bb_width`: VWMA-based Bollinger Band metrics

### 🌍 Market Index Indicators
- `kospi_close_rate`, `kosdaq_close_rate`
- `kospi_vwma5_gap`, `kospi_vwma20_gap`
- `kosdaq_vwma5_gap`, `kosdaqi_vwma20_gap`
- `sp500f_close_rate`, `sp500f_ma5_gap`, `sp500f_ma20_gap`
- `nasdaq100f_close_rate`, `nasdaq100f_ma5_gap`, `nasdaq100f_ma20_gap`
- `vix_close_rate`, `sp500v_close_rate`

### 🔁 Volume & Flow Indicators
- `trading_volume_volatility_ratio`
- `trading_change`, `trading_rolling_change`
- `foreign_rate`, `institution_rate`, `individual_rate`
- `foreign_net_buy_days`, `institution_net_buy_days`

### 📈 Candlestick Patterns
- `candle_upper_tail_ratio`, `candle_lower_tail_ratio`
- `candle_body_ratio`, `candle_sign`

## Feature Heatmap
The following heatmap shows pairwise correlations between selected input features.

![heatmap (2)](https://github.com/user-attachments/assets/00df943a-1b7e-40e1-a051-867734e59d49)

## Distribution per Feature
Below are the individual distributions of input features used in model training.
|  |  |  |  |
|--|--|--|--|
| ![is_kospi](https://github.com/user-attachments/assets/792e12c2-18a5-4510-a2b4-3bc625395e2a) | ![close_rate](https://github.com/user-attachments/assets/33ffa740-deb6-402b-bff1-7987fcc633e4) | ![vwma5_gap](https://github.com/user-attachments/assets/4f7337dd-8ea2-4ccc-b29f-bfdb3380742e) | ![vwma20_gap](https://github.com/user-attachments/assets/156dd8c7-4285-487c-9a0d-21b98c93c906) | 
| ![vwma_bb_upper_ratio](https://github.com/user-attachments/assets/f5f9837d-a47e-49f0-8c75-cc1cde26fe46) | ![vwma_bb_lower_ratio](https://github.com/user-attachments/assets/a84162a2-b220-41d0-ac66-c33b0dc5c7e8) | ![vwma_bb_width](https://github.com/user-attachments/assets/af0529c2-769c-4144-9e8c-77e15075a88f) | ![kospi_vwma5_gap](https://github.com/user-attachments/assets/91c21f1d-aced-4f83-897d-eea6ba8b112b) | 
| ![kospi_vwma20_gap](https://github.com/user-attachments/assets/aa361acc-43cf-460d-9746-6b0b0ae37dd7) | ![kosdaq_vwma5_gap](https://github.com/user-attachments/assets/e0492513-b559-4252-bf33-cb0a90e85ad2) | ![kosdaq_vwma20_gap](https://github.com/user-attachments/assets/a0b89e95-c223-492e-8e57-484729e7d9a3) | ![open_to_close](https://github.com/user-attachments/assets/d51f6627-4954-4819-adbd-26049ae3e3b6) | 
| ![high_to_low](https://github.com/user-attachments/assets/0c9227c6-382d-4313-bc93-ea48f7ade250) | ![rsi](https://github.com/user-attachments/assets/cf91a673-10a3-445c-8e42-1e1f4667dd34) | ![macd_ratio](https://github.com/user-attachments/assets/b412ee5f-eac5-40c0-b1ad-0e6207528bbf) | ![macd_signal_ratio](https://github.com/user-attachments/assets/42141dca-b4c7-4a36-8e4d-32882cfaa7a2) | 
| ![macd_golden_cross](https://github.com/user-attachments/assets/7b9e0df4-e4ed-42a6-88a1-9e35d3271023) | ![macd_dead_cross](https://github.com/user-attachments/assets/f3645df7-eae2-4c0d-a831-3409d4e8b74e) | ![atr_ratio](https://github.com/user-attachments/assets/bc942118-0526-4bf5-bba0-a9bc776eded5) | ![kospi_close_rate](https://github.com/user-attachments/assets/1ed5cebd-60b2-4a09-8aeb-33cc1f125dec) | 
| ![kosdaq_close_rate](https://github.com/user-attachments/assets/8a06c2b9-f9b4-43fc-ab38-c17533253197) | ![sp500f_close_rate](https://github.com/user-attachments/assets/e3f0791c-80e7-43ce-8266-8c6dfdbce076) | ![sp500f_ma5_gap](https://github.com/user-attachments/assets/1f2fa8e7-ba72-4c85-8ad8-d387c308c596) | ![sp500f_ma20_gap](https://github.com/user-attachments/assets/835db012-3021-44fb-8926-0c02f09f287a) | 
| ![nasdaq100f_close_rate](https://github.com/user-attachments/assets/adc0fc5f-5878-4be0-b517-70190e2cfa43) | ![nasdaq100f_ma5_gap](https://github.com/user-attachments/assets/f697aa28-46f1-4bf2-b44b-aeac0106124d) | ![nasdaq100f_ma20_gap](https://github.com/user-attachments/assets/ad64c62d-adc0-45fd-bab6-296b9744dcf9) | ![vix_close_rate](https://github.com/user-attachments/assets/34a69461-9480-4532-a8dc-488609e87e67) | 
| ![sp500v_close_rate](https://github.com/user-attachments/assets/ffb5ec1a-452f-467e-a8ff-76925d3d92f8) | ![trading_volume_volatility_ratio](https://github.com/user-attachments/assets/532edee0-7dbf-4aab-bb94-c0af60e7eb12) | ![trading_change](https://github.com/user-attachments/assets/0ca135f0-0484-4927-ba1e-0df332a28a6d) | ![trading_rolling_change](https://github.com/user-attachments/assets/d527377c-7eda-439b-b612-b117d3f2e3d0) | 
| ![foreign_rate](https://github.com/user-attachments/assets/49a70ad6-f283-41ca-98eb-3eee0cd8d52f) | ![institution_rate](https://github.com/user-attachments/assets/7dda2579-2285-4ba8-a85d-13ce2e40aa6a) | ![individual_rate](https://github.com/user-attachments/assets/16a34c2b-214f-4667-8533-becf7ce6f830) | ![foreign_net_buy_days](https://github.com/user-attachments/assets/8298aaa9-8462-4e11-8fd2-16be299067b1) | 
| ![institution_net_buy_days](https://github.com/user-attachments/assets/3f7b75a6-0fd1-4397-a979-cf04cc846c8d) | ![candle_upper_tail_ratio](https://github.com/user-attachments/assets/686e45ec-d450-4a8d-b14a-bd017c775d7b) | ![candle_lower_tail_ratio](https://github.com/user-attachments/assets/059314ef-b958-4c1c-a1c5-5807d0b408b4) | ![candle_body_ratio](https://github.com/user-attachments/assets/62ff6aed-9915-47ba-9d1a-bb2b284db7c3) | 
| ![candle_sign](https://github.com/user-attachments/assets/99817acb-76f1-46b4-8742-3a25f782c81c) | | |

---

## 🔧 Key Features
- Patch-wise Transformer encoder for time-series inputs
- Industry embedding integration (mid-level industry classification)
- Soft label generation from 5-day future returns
- LambdaRankLoss for pairwise ranking optimization
- Real-time applicability:  
  Designed for inference and trade execution near market close (15:40–16:00), based on the final closing price.

## 🎯 Target Selection Strategy
- Top 200 stocks by daily trading volume
- Market cap ≥ 500B KRW
- Excludes limit-up and newly listed stocks

## 🔧 Model Hyperparameters
- Batch Size: 1 (up to 200 stocks grouped per day)
- Input Dimension: 41
- Industry Embedding Dimension: 4
- Model Dimension: 64
- Sliding Window Size: 30
- Patch Length: 6
- Patch Stride: 3
- Number of Attention Heads: 4
- Number of Transformer Encoder Layers: 2
- Dropout: 2.5
- Learning Rate: 5e-4
- Weight Decay: 5e-5
- Number of Epochs: 100
- Early Stopping Patience: 10

## 🏷️ Labeling & Ranking
- 5-day return quintiles (20 bins) used for soft labels
- TOP3 selection by predicted score per day

### 🧮 20-Quantile Label Bins
|  |  |  |
|--|--|--|
| ![training_labels_20](https://github.com/user-attachments/assets/f0a7dee6-ef76-419a-ae4a-13dbd81f8ec8) | ![validation_labels_20](https://github.com/user-attachments/assets/65575d5c-0db2-47ed-970d-b80b6ce21c61) | ![test_labels_20](https://github.com/user-attachments/assets/badd344f-84ae-4d46-b117-cf45beab74dc) |

### 📉 Distribution of Predicted Scores
|  |  |
|--|--|
| ![val_top3_pred (2)](https://github.com/user-attachments/assets/494a7665-65c9-4286-a4b7-d59ade67a2c5) | ![test_top3_pred (2)](https://github.com/user-attachments/assets/c2725ab4-1a26-4dce-af85-f4d931e8527f) |

### 🔍 Label Distribution
|  |  |  |
|--|--|--|
| ![training_labels](https://github.com/user-attachments/assets/aa33dd41-d322-44ac-9b70-79ba09b61491) | ![validation_labels](https://github.com/user-attachments/assets/0da18d61-a172-428a-9b5d-1faf512ba99e) | ![test_labels](https://github.com/user-attachments/assets/ac014808-9aee-4a13-8979-8083985cca74) |

### 📚 Learn More: Ranking Metrics & Loss

For an in-depth explanation of the ranking metric and training objective used in this model, see:

- 📘 [NDCG Explained](docs/NDCG.md):  
  Understand how Normalized Discounted Cumulative Gain (NDCG) measures ranking quality in stock selection.

- ⚙️ [LambdaRank Loss Guide](docs/LAMBDA_RANK_LOSS.md):  
  Dive into the pairwise ranking loss function that optimizes NDCG by comparing stock relevance in every batch.

---

## 📈 Return Evaluation
| Item                          | 2024 TOP3                       | 2025 TOP3                       |
|-------------------------------|----------------------------------|----------------------------------|
| Number of Samples             | 195,866                         | 42,966                          |
| Number of Buy / Sell Trades   | 173 / 173                       | 27 / 27                         |
| **Win Rate (Count, Ratio)**   | 96 trades (55.49%)            | 22 trades (81.48%)             |
| **Loss Rate (Count, Ratio)**  | 77 trades (44.51%)             | 5 trades (18.52%)             |
| Average Return (Win)          | 6.66%                           | 8.78%                           |
| Average Return (Loss)         | -4.90%                          | -4.49%                          |
| Avg. Holding Period (Win)     | 9.9 calendar days (6.5 trading days) | 9.1 calendar days (6.1 trading days) |
| Avg. Holding Period (Loss)    | 9.9 calendar days (6.7 trading days) | 13.6 calendar days (7.2 trading days) |
| Return Deciles                | [-28.7, -6.5, -3.5, -2.2, -0.5, 1.0, 2.5, 4.9, 6.9, 9.7, 34.6] | [-8.2, -3.1, 0.9, 1.2, 3.0, 4.0, 5.6, 7.6, 11.5, 21.2, 30.7] |
| Trade Capital                 | 10,000,000                     | 10,000,000                     |
| **Expected Net Return**       | **1.515%**                      | **6.323%**                      |
| **Cumulative Net Profit**     | **26,210,875**                  | **17,072,862**                  |

### 📅 Monthly Return
![val_month_roi (2)](https://github.com/user-attachments/assets/5c22536a-050f-45fc-9cf4-9f7b9a42050b)
![test_month_roi (2)](https://github.com/user-attachments/assets/bf980f7e-c9bd-448e-9e3d-77777581e1aa)

### 📊 Daily Return
![val_january_roi (2)](https://github.com/user-attachments/assets/28d74668-f108-418e-851e-6e85351ac4e3)
![test_january_roi (2)](https://github.com/user-attachments/assets/fdb8b77f-a4b4-4146-a42b-cde6eb1312cd)

### 🏦 Return by Stock
![val_stocks_roi (2)](https://github.com/user-attachments/assets/b716c1fe-7d3e-4cda-8e1d-5ac9a1e907a9)

### 🤔 Return Distribution by Purchase Decision
|  |  |
|--|--|
| ![매수에_따라_n일_후_수익률_평균값 (2)](https://github.com/user-attachments/assets/fda20527-be63-4c29-93bc-03fd709e0925) | ![매수에_따라_n일_후_수익률_분포 (2)](https://github.com/user-attachments/assets/3fbe2ae7-9203-432b-8d46-322b192f094b) |
| ![매수_False_일_때_n일_후_수익률_분포 (2)](https://github.com/user-attachments/assets/c6025ef4-8298-4154-bc58-cc617f5148b1) | ![매수_True_일_때_n일_후_수익률_분포 (2)](https://github.com/user-attachments/assets/ac17605d-270b-45d8-be12-4e17d48d2c2d) |

## 🔍 Case Study: Specific Stocks
![val_한화오션](https://github.com/user-attachments/assets/106851f3-8a99-4c2c-828f-2cf967bd148a)

---

## 💻 Environment

- Python 3.12.8
- PyTorch 2.6.0 + CUDA 12.6  
- pykrx 1.0.48  
- See [requirements.txt](./requirements.txt) for full dependency list

## 🧪 Experiment Notebook

The full end-to-end workflow is implemented in the following notebook:<br>
[Stock_PatchTST_Ranking.ipynb](./Stock_PatchTST_Ranking.ipynb)

This includes:
- Raw data retrieval
- Feature engineering and preprocessing
- Model training and validation
- Return evaluation and analysis
- Case studies on selected stocks

---

## 📄 License & Acknowledgements

This project is licensed under the MIT License.  
See the [LICENSE](./LICENSE) file for details.

This work is inspired by [PatchTST](https://github.com/yuqinie98/PatchTST).  
It was developed and tested on KRX daily stock data from 2020 to 2025.
KRX data was primarily retrieved using the [pykrx](https://github.com/sharebook-kr/pykrx) library.  
Industry classification codes were retrieved via the Open API from [Korea Investment & Securities](https://github.com/koreainvestment/open-trading-api).


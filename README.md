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
📘 See [TRANSFORMER.md](docs/TRANSFORMER.md) for a detailed architecture explanation.

![stock_patch_tst_model](https://github.com/user-attachments/assets/0d4c28c7-1bc2-4ea8-93cb-51896741d3c6)
> Patches are extracted via Conv1D to compress temporal data, and a Transformer encoder captures inter-patch dependencies for ranking prediction.

---

## 🧠 Input Features
📘 Full feature list: [FEATURES_TECHNICAL_INDICATORS.md](docs/FEATURES_TECHNICAL_INDICATORS.md)

- **Feature engineering:**<br>
Ratio-based transformations, log-scaling for long-tail features, stable across different stocks.

### 🏭 Stock Metadata
- `industry_id`, `is_kospi`

### 💹 Price Flow Indicators
- `close_rate`, `open_to_close`, `high_to_low`, `rsi`, `ato`, `macd` related

### 📊 VWMA & Bollinger Bands
- `vwma5_gap`, `vwma20_gap`, `vwma_bb_width` etc.

### 🌍 Market Indices
- KOSPI, KOSDAQ, S&P500 Futures, Nasdaq 100 Futures, VIX, etc.

### 🔁 Volume & Flow
- Volume volatility ratios, net buy rates

### 📈 Candlestick Patterns
- Upper/lower tail ratios, body ratios

## Feature Heatmap
The following heatmap shows pairwise correlations between selected input features.
![heatmap](https://github.com/user-attachments/assets/d91c5728-0bc5-4b72-8a17-26e8d5f5fe5e)

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
- Patch-wise Transformer encoder
- Industry embedding
- Soft label generation from 5-day future returns
- LambdaRankLoss for ranking optimization
- Real-time applicability (15:40–16:00 trading window)

## 🎯 Target Selection Strategy
- Top 200 stocks by daily trading volume
- Market cap ≥ 500B KRW
- Excludes limit-up and newly listed stocks

## 🔧 Model Hyperparameters
- Input Dim: 41
- Industry Embedding Dim: 4
- Model Dim: 64
- Sliding Window Size: 30
- Patch Length: 6, Stride: 3
- Transformer Encoder Heads: 4, Layers: 2
- Dropout: 2
- Learning Rate: 5e-4
- Weight Decay: 5e-5
- Early Stopping Patience: 10 epochs

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
| ![val_top3_pred](https://github.com/user-attachments/assets/f7ba6a34-25fb-4369-8e88-9d6339f64773) | ![test_top3_pred](https://github.com/user-attachments/assets/2c383056-2926-4f43-a4dc-aaa42fb43fcd) |

### 🔍 Raw Label Distribution
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

## 🔍 Post-Filtering Rules
While model prediction provides initial candidates, additional filtering and dynamic sell strategies are applied to make the system robust for real-world trading.

### 🛒 Buy Signal Post-Filtering
- `pred_rank <= topn` (e.g., TOP3)
- `atr_ratio > 0.03` (sufficient volatility)
- `close_rate > -10%` (avoiding sharp decliners)

---

### 💵 Sell Signal Detection

- **Max holding period**: Forced exit after fixed days (e.g., 5 days).
- **Trailing Entry Extension**:<br>
If a new buy signal occurs during holding, reset holding period.

---

## 📈 Return Evaluation
| Item                          | 2024 TOP3                                                      | 2025 TOP3                                                     |
|-------------------------------|----------------------------------------------------------------|---------------------------------------------------------------|
| Number of Samples             | 195,866                                                        | 42,966                                                        |
| Number of Buy / Sell Trades   | 205 / 205                                                      | 35 / 35                                                       |
| **Win Rate (Count, Ratio)**   | 112 trades (54.63%)                                            | 26 trades (74.29%)                                            |
| **Loss Rate (Count, Ratio)**  | 93 trades (45.37%)                                             | 9 trades (25.71%)                                             |
| Average Return (Win)          | 6.23%                                                          | 7.54%                                                         |
| Average Return (Loss)         | -4.57%                                                         | -3.88%                                                        |
| Avg. Holding Period (Win)     | 9.6 calendar days (6.4 trading days)                           | 9.2 calendar days (5.8 trading days)                          |
| Avg. Holding Period (Loss)    | 9.5 calendar days (6.5 trading days)                           | 10.7 calendar days (6.2 trading days)                         |
| Return Deciles                | [-36.0, -6.5, -3.6, -2.0, -0.7, 0.8, 2.1, 4.1, 5.9, 9.9, 34.6] | [-8.2, -3.8, -2.3, 0.5, 1.1, 2.5, 4.2, 7.2, 8.3, 16.3, 30.7] |
| Trade Capital                 | 10,000,000                                                     | 10,000,000                                                    |
| **Expected Net Return**       | **1.335%**                                                     | **4.601%**                                                    |
| **Cumulative Net Profit**     | **27,365,419**                                                 | **16,105,142**                                                |

### 📅 Monthly Return
![2024수익](https://github.com/user-attachments/assets/0dd992cb-b1fe-4296-ba36-2934ed48c20c)
![2025수익](https://github.com/user-attachments/assets/e42a9a94-42c2-4af5-9ed8-6e7e26fc71d9)

### 📊 Daily Return
![202401수익](https://github.com/user-attachments/assets/40682580-7101-4bb2-ae72-d10ded5c8f8a)
![202501수익](https://github.com/user-attachments/assets/80012f30-8028-49cc-82b7-eb9e9a19a884)

### 🏦 Return by Stock
![종목별수익](https://github.com/user-attachments/assets/a2bd444a-07a5-48a1-992b-3c4cf2488be6)

### 🤔 Return Distribution by Purchase Decision
|  |  |
|--|--|
| ![수익률평균값](https://github.com/user-attachments/assets/63e7a7b5-9854-4d69-bae7-8e5feef9b0a2) | ![수익률분포](https://github.com/user-attachments/assets/152c4dba-d057-4daa-a55e-9e3cbe56a7a8) |
| ![수익률분포0](https://github.com/user-attachments/assets/ccddad30-aac4-4e48-b7b9-b55921c9d4e8) | ![수익률분포1](https://github.com/user-attachments/assets/30d1b764-5ad3-4c36-88b1-fba2fd7d8ad9) |

## 🔍 Case Study: Specific Stocks
![한화오션](https://github.com/user-attachments/assets/0a467d36-5e75-447f-a279-07a291e0d0f9)

---

## 💻 Environment

- Python 3.12.8
- PyTorch 2.6.0 + CUDA 12.6  
- pykrx 1.0.48  
- Full list in [requirements.txt](./requirements.txt)

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


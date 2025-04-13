![StockPatchTST_banner_proportional_1024x400](https://github.com/user-attachments/assets/9470ab1a-42c3-4ee7-9961-08d627b9a0d7)
![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.6.0-%23ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

## 📚 Table of Contents
- [Input Features](#-input-features)
- [Key Features](#-key-features)
- [Target Selection Strategy](#-target-selection-strategy)
- [Training Setup](#%EF%B8%8F-training-setup)
- [Labeling & Ranking](#%EF%B8%8F-labeling--ranking)
- [Return Evaluation](#-return-evaluation)
- [Case Study](#-case-study-specific-stocks)
- [Environment](#-environment)
- [Experiment Notebook](#-experiment-notebook)
- [License](#-license--acknowledgements)

## Overview of Deep Learning Model Evolution
A chronological overview of model improvements, from LSTM to Transformer variants.

![model_history](https://github.com/user-attachments/assets/7795238a-b8e8-40f5-90c7-034a854555b3)

## Transformer-Based Ranking Model for Stock Selection
![stock_patch_tst](https://github.com/user-attachments/assets/11c5425c-683d-4d3a-bc50-397ee36b5aed)

---

## 🧠 Input Features
The model was trained using a wide range of features including price movement, volatility, volume trends, market indices, and candlestick patterns.  
Below is the list of major features used for training.

> 💡 **Note on Feature Engineering**  
> All input features were transformed into **ratio-based values** to enhance generalizability across different price/volume levels.  
> Instead of applying sliding-window or historical normalization, features were expressed as relative changes (e.g., % gap from moving averages, rate-of-change, ratios).  
> Long-tail distributions (e.g., volume volatility ratios) were handled via simple log-scaling.  
> This approach was chosen to keep the feature space interpretable and stable across different stocks and timeframes.

### 📘 [Feature Definitions & Formulas](./FEATURES_TECHNICAL_INDICATORS.md)

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

![heatmap](https://github.com/user-attachments/assets/12db48fa-fa1c-4729-a20b-3d14c0d92691)

## Distribution per Feature
Below are the individual distributions of input features used in model training.
|  |  |  |  |
|--|--|--|--|
![is_kospi](https://github.com/user-attachments/assets/f8dfb31b-f203-46d6-ae0a-703c11ea94a5) | ![close_rate](https://github.com/user-attachments/assets/858dcfe0-8c8f-4c24-bbe0-bdf9dc874a7d) | ![vwma5_gap](https://github.com/user-attachments/assets/5301b2af-7a0d-4b4f-9907-a34473b8fd96) | ![vwma20_gap](https://github.com/user-attachments/assets/895bfe69-8e3c-4a92-b901-69813c05da7e)
![vwma_bb_upper_ratio](https://github.com/user-attachments/assets/a22070ec-7033-4008-8b66-3d8451649060) | ![vwma_bb_lower_ratio](https://github.com/user-attachments/assets/c5d52757-7453-4070-8cca-68788b44d835) | ![vwma_bb_width](https://github.com/user-attachments/assets/bbf9711b-55f6-4b42-83e6-35efcc942219) | ![kospi_vwma5_gap](https://github.com/user-attachments/assets/41fe3219-08dd-4a7b-96e9-6bc58fd78e13)
![kospi_vwma20_gap](https://github.com/user-attachments/assets/718cf0a5-41aa-4e41-823f-9ca9814194a0) | ![kosdaq_vwma5_gap](https://github.com/user-attachments/assets/b8d05399-099d-4a44-bfbf-8ce51a36cb35) | ![kosdaq_vwma20_gap](https://github.com/user-attachments/assets/5470bc29-17ac-4f53-b0c2-415f190828bf) | ![open_to_close](https://github.com/user-attachments/assets/b68bda2c-8e14-4488-a013-389dcb4c76f8)
![high_to_low](https://github.com/user-attachments/assets/33ba1fe3-0bd8-47ea-86d9-a32843498fce) | ![rsi](https://github.com/user-attachments/assets/71170548-a293-4352-8439-5809e3898b2e) | ![macd_ratio](https://github.com/user-attachments/assets/7f98f4e1-b7cb-4fcd-865d-101af5df61a7) | ![macd_signal_ratio](https://github.com/user-attachments/assets/31edb281-2587-4755-9bf3-f79401f54f20)
![macd_golden_cross](https://github.com/user-attachments/assets/70b6fa35-23f2-4308-9027-29724094c150) | ![macd_dead_cross](https://github.com/user-attachments/assets/2c189d3f-65da-4d26-b5c0-73da160c3afb) | ![atr_ratio](https://github.com/user-attachments/assets/0004fe80-d61b-48b8-bd72-a4c03c7f7ff9) | ![kospi_close_rate](https://github.com/user-attachments/assets/994a6c5d-96a5-43e9-a898-0b77c44ea381)
![kosdaq_close_rate](https://github.com/user-attachments/assets/04b527f5-eb55-4408-8754-56c1302f9917) | ![sp500f_close_rate](https://github.com/user-attachments/assets/2c55e0e0-efc0-4ed6-901d-f253ece8d84a) | ![sp500f_ma5_gap](https://github.com/user-attachments/assets/2218a12b-ec33-4bed-897e-0ce6a65fe7f2) | ![sp500f_ma20_gap](https://github.com/user-attachments/assets/f47861ad-1205-4895-9e96-870a8656b796)
![nasdaq100f_close_rate](https://github.com/user-attachments/assets/62cabd43-a2a1-4315-924b-30c359ba9b6d) | ![nasdaq100f_ma5_gap](https://github.com/user-attachments/assets/27665347-f9f0-40e4-8f22-b55163c374f3) | ![nasdaq100f_ma20_gap](https://github.com/user-attachments/assets/941c639e-15a7-4e8c-aed2-6e74b4100ae9) | ![vix_close_rate](https://github.com/user-attachments/assets/13b9c112-b88e-403b-a3c2-1b547cd86834)
![sp500v_close_rate](https://github.com/user-attachments/assets/10681a42-474c-4e9b-836b-0ca257779662) | ![trading_volume_volatility_ratio](https://github.com/user-attachments/assets/e7a349ce-7645-45f8-96a0-f9fd6ffd3e04) | ![trading_change](https://github.com/user-attachments/assets/5257edc2-fd0e-46a2-96af-df1218d1c79b) | ![trading_rolling_change](https://github.com/user-attachments/assets/d5aba883-5db4-4aae-a3e3-51e2aae869b4)
![foreign_rate](https://github.com/user-attachments/assets/005c3db4-a186-49c7-a043-068d0739362d) | ![institution_rate](https://github.com/user-attachments/assets/78cb107f-2feb-4c5e-9219-5315dfdcb5cf) | ![individual_rate](https://github.com/user-attachments/assets/2ea09e43-c2a2-4cd8-9bc8-9323f23876df) | ![foreign_net_buy_days](https://github.com/user-attachments/assets/7076214f-f9f9-462b-85e2-34e009f6e3a5)
![institution_net_buy_days](https://github.com/user-attachments/assets/f1bc7213-5047-4dc0-b754-627002236057) | ![candle_upper_tail_ratio](https://github.com/user-attachments/assets/6a747311-55b2-46af-a470-8a24e1879b86) | ![candle_lower_tail_ratio](https://github.com/user-attachments/assets/1c6acc0a-6220-47ee-954a-506b51c21fc6) | ![candle_body_ratio](https://github.com/user-attachments/assets/9e4196ae-6070-428a-aae1-60887fccab02)
![candle_sign](https://github.com/user-attachments/assets/514102a9-6dbf-4346-946c-4af1f30481c1) |  |  |  

---

## 🔧 Key Features
- Patch-wise Transformer encoder for time-series inputs
- Industry embedding integration (mid-level industry classification)
- Soft label generation from 5-day future returns
- LambdaRankLoss for pairwise ranking optimization
- Top-volume filtering & large-cap only:  
  Model selects stocks within the top 200 by daily trading volume and market capitalization of at least 500B KRW.
- Real-time applicability:  
  Designed for inference and trade execution near market close (15:40–16:00), based on the final closing price.

## 🎯 Target Selection Strategy
- Top 200 stocks by daily trading volume
- Market cap ≥ 500B KRW
- Trades executed at closing auction (15:40–16:00)
- Excludes limit-up and newly listed stocks

## ⚙️ Training Setup
- Dropout: 2.0  
- Learning Rate: 5e-4  
- Weight Decay: 5e-5  
- Batch-wise stock grouping (up to 200 per day)

## 🏷️ Labeling & Ranking
- 5-day return quintiles (5 bins) used for soft labels
- TOP3 selection by predicted score per day

### 🧮 Five-Quantile Label Bins
|  |  |  |
|--|--|--|
| ![training_labels_5](https://github.com/user-attachments/assets/595a8fd1-9f43-419c-b7ce-a4c054e9dc25) | ![validation_labels_5](https://github.com/user-attachments/assets/d6af6c7f-31b8-4af5-bdea-2a492cf2c9c7) | ![test_labels_5](https://github.com/user-attachments/assets/aec88eac-b583-4e41-bff8-b7f372cbc5cb) |

### 📉 Distribution of Predicted Scores
|  |  |
|--|--|
| ![val_top3_pred](https://github.com/user-attachments/assets/12c54e2f-2d06-4027-88b5-7f907e7b89bb) | ![test_top3_pred](https://github.com/user-attachments/assets/f89126b3-bacb-4286-ab35-339d1856299b) |

### 🔍 Label Distribution
|  |  |  |
|--|--|--|
| ![training_labels](https://github.com/user-attachments/assets/42283e32-5139-4f1e-82c9-11ca006db625) | ![validation_labels](https://github.com/user-attachments/assets/bb5147a8-42bb-41e0-bdff-b95e87c416f7) | ![test_labels](https://github.com/user-attachments/assets/d873d458-7bd8-4e3a-8cac-a07433a706d1) |

### 📚 Learn More: Ranking Metrics & Loss

For an in-depth explanation of the ranking metric and training objective used in this model, see:

- 📘 [NDCG Explained](./NDCG.md):  
  Understand how Normalized Discounted Cumulative Gain (NDCG) measures ranking quality in stock selection.

- ⚙️ [LambdaRank Loss Guide](./LAMBDA_RANK_LOSS.md):  
  Dive into the pairwise ranking loss function that optimizes NDCG by comparing stock relevance in every batch.

---

## 📈 Return Evaluation
| Item                          | 2024 TOP3                       | 2025 TOP3                       |
|-------------------------------|----------------------------------|----------------------------------|
| Number of Samples             | 191,718                         | 42,082                          |
| Number of Buy / Sell Trades   | 201 / 201                       | 38 / 38                         |
| **Win Rate (Count, Ratio)**   | 105 trades (52.24%)            | 27 trades (71.05%)             |
| **Loss Rate (Count, Ratio)**  | 96 trades (47.76%)             | 11 trades (28.95%)             |
| Average Return (Win)          | 7.07%                           | 7.99%                           |
| Average Return (Loss)         | -4.50%                          | -3.56%                          |
| Avg. Holding Period (Win)     | 9.9 calendar days (6.6 trading days) | 9.7 calendar days (6.3 trading days) |
| Avg. Holding Period (Loss)    | 9.9 calendar days (6.7 trading days) | 10.2 calendar days (6.2 trading days) |
| Return Deciles                | [-36.0, -7.1, -3.5, -1.7, -0.7, 0.5, 2.3, 4.1, 5.9, 9.2, 36.3] | [-8.2, -4.2, -1.7, 0.1, 0.9, 2.2, 4.0, 7.3, 8.7, 19.8, 30.7] |
| Trade Capital                 | 10,000,000                     | 10,000,000                     |
| **Expected Net Return**       | **1.544%**                      | **4.646%**                      |
| **Cumulative Net Profit**     | **31,043,339**                  | **17,653,692**                  |

### 📅 Monthly Return
![val_month_roi](https://github.com/user-attachments/assets/f9f10013-53a5-46c8-bc3b-5b505599d2f3)
![test_month_roi](https://github.com/user-attachments/assets/c6ae1c82-3ec4-40c2-bed1-c613e077829f)

### 📊 Daily Return
![val_january_roi](https://github.com/user-attachments/assets/ca235229-06b9-49a6-b6f9-1f0a1d5f95a0)
![test_january_roi](https://github.com/user-attachments/assets/7bd4d324-af93-4866-87ce-cc21e6d6ec74)

### 🏦 Return by Stock
![val_stocks_roi](https://github.com/user-attachments/assets/03571f4b-7025-4e2c-8934-e1550781ea86)

### 🤔 Return Distribution by Purchase Decision
|  |  |
|--|--|
| ![매수에_따라_n일_후_수익률_평균값](https://github.com/user-attachments/assets/13b0e320-afd8-4796-a1ee-7bcab90b49a0) | ![매수에_따라_n일_후_수익률_분포](https://github.com/user-attachments/assets/cb8d4c35-864a-41a1-a203-d45a62227d25) |
| ![매수_False_일_때_n일_후_수익률_분포](https://github.com/user-attachments/assets/223edd3f-977d-4283-bcd9-91016c2bf83d) | ![매수_True_일_때_n일_후_수익률_분포](https://github.com/user-attachments/assets/c1b5f3be-62a5-42ab-80a5-070db23bece9) |

## 🔍 Case Study: Specific Stocks
![val_한화시스템](https://github.com/user-attachments/assets/6a1f98ad-6d76-4720-a053-8ae578b6dbeb)
![val_HLB제약](https://github.com/user-attachments/assets/9c41a246-f827-41d7-9581-cbf56f95a76b)

---

## 💻 Environment

- Python 3.12.8
- PyTorch 2.6.0 + CUDA 12.6  
- pykrx 1.0.48  
- See [requirements.txt](./requirements.txt) for full dependency list

## 🧪 Experiment Notebook

The full end-to-end workflow is implemented in the following notebook:<br>
[Stock_TimeSeriesTransformer_Ranking.ipynb](./Stock_TimeSeriesTransformer_Ranking.ipynb)

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


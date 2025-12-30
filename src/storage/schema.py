import pyarrow as pa

# Raw Crypto OHLCV 5m
RAW_CRYPTO_OHLCV_5M = pa.schema([
    ("timestamp_utc", pa.timestamp('ms')), # UTC timestamp
    ("exchange_code", pa.int8()),          # 1=Binance, 2=Bybit
    ("symbol", pa.string()),
    ("market_kind", pa.string()),          # spot, perp
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("volume", pa.float64()),
    ("quote_volume", pa.float64()),
    ("trade_count", pa.int64()),
    ("ingested_at_utc", pa.timestamp('ms')),
    ("source", pa.string()),
    ("year_month", pa.string())
])

# Raw Crypto BookTicker
RAW_CRYPTO_BOOKTICKER = pa.schema([
    ("timestamp_utc", pa.timestamp('ms')),
    ("exchange_code", pa.int8()),
    ("symbol", pa.string()),
    ("bid", pa.float64()),
    ("ask", pa.float64()),
    ("ingested_at_utc", pa.timestamp('ms')),
    ("source", pa.string()),
    ("year_month", pa.string())
])

# Raw Crypto AggTrades
RAW_CRYPTO_AGGTRADES = pa.schema([
    ("timestamp_utc", pa.timestamp('ms')),
    ("exchange_code", pa.int8()),
    ("symbol", pa.string()),
    ("price", pa.float64()),
    ("qty", pa.float64()),
    ("is_buyer_maker", pa.bool_()),
    ("ingested_at_utc", pa.timestamp('ms')),
    ("source", pa.string()),
    ("year_month", pa.string())
])

# Raw Crypto Derivatives
RAW_CRYPTO_DERIVATIVES = pa.schema([
    ("timestamp_utc", pa.timestamp('ms')),
    ("exchange_code", pa.int8()),
    ("symbol", pa.string()),
    ("funding_rate", pa.float64()),
    ("open_interest", pa.float64()),
    ("liquidations_long", pa.float64()),
    ("liquidations_short", pa.float64()),
    ("ingested_at_utc", pa.timestamp('ms')),
    ("source", pa.string()),
    ("year_month", pa.string())
])

# Raw Equities 1d
RAW_EQUITIES_1D = pa.schema([
    ("date_utc", pa.date32()),
    ("ticker", pa.string()),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("volume", pa.float64()),
    ("ingested_at_utc", pa.timestamp('ms')),
    ("source", pa.string()),
    ("year_month", pa.string())
])

SCHEMAS = {
    "raw_crypto_ohlcv_5m": RAW_CRYPTO_OHLCV_5M,
    "raw_crypto_bookticker": RAW_CRYPTO_BOOKTICKER,
    "raw_crypto_aggtrades": RAW_CRYPTO_AGGTRADES,
    "raw_crypto_derivatives": RAW_CRYPTO_DERIVATIVES,
    "raw_equities_1d": RAW_EQUITIES_1D,
    "aligned_crypto_5m": pa.schema([
        ("timestamp_utc", pa.timestamp('ms')),
        ("exchange_code", pa.int8()),
        ("symbol", pa.string()),
        ("market_kind", pa.string()),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("volume", pa.float64()),
        ("quote_volume", pa.float64()),
        ("trade_count", pa.int64()),
        ("ingested_at_utc", pa.timestamp('ms')),
        ("source", pa.string()),
        ("is_missing", pa.bool_()),
        ("year_month", pa.string())
    ]),
    "aligned_equities_1d": pa.schema([
        ("date_utc", pa.timestamp('ms')),
        ("ticker", pa.string()),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("volume", pa.float64()),
        ("ingested_at_utc", pa.timestamp('ms')),
        ("source", pa.string()),
        ("is_missing", pa.bool_()),
        ("year_month", pa.string())
    ]),
    "patch_crypto_1h": pa.schema([
        ("timestamp_utc", pa.timestamp('ms')),
        ("symbol", pa.string()),
        ("RET", pa.float64()),
        ("RVOL", pa.float64()),
        ("VLM", pa.float64()),
        ("FLOW", pa.float64()),
        ("SPREAD", pa.float64()),
        ("OI", pa.float64()),
        ("FUND", pa.float64()),
        ("BASIS", pa.float64()),
        ("exchange_code", pa.int8()),
        ("market_kind", pa.string())
    ]),
    "patch_equities_5d": pa.schema([
        ("date_utc", pa.timestamp('ms')),
        ("ticker", pa.string()),
        ("RET", pa.float64()),
        ("RVOL", pa.float64()),
        ("VLM", pa.float64()),
        ("FLOW", pa.float64()),
        ("SPREAD", pa.float64()),
        ("OI", pa.float64()),
        ("FUND", pa.float64()),
        ("BASIS", pa.float64())
    ]),
    "events_crypto": pa.schema([
        ("timestamp_utc", pa.timestamp('ms')),
        ("symbol", pa.string()),
        ("event_type", pa.string()),
        ("event_value", pa.float64()),
        ("direction", pa.string()),
        ("confidence", pa.float64())
    ]),
    "events_equities": pa.schema([
        ("timestamp_utc", pa.timestamp('ms')),
        ("ticker", pa.string()),
        ("event_type", pa.string()),
        ("event_value", pa.float64()),
        ("direction", pa.string()),
        ("confidence", pa.float64())
    ])
}

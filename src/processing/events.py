import pandas as pd
import numpy as np
import logging
from pathlib import Path
from src.storage.writer import save_partitioned

logger = logging.getLogger("processing.events")

class EventGenerator:
    def __init__(self, config):
        self.config = config
        self.patch_path = Path(config["paths"]["patch"])
        self.events_path = Path("data/events")
        self.events_path.mkdir(parents=True, exist_ok=True)
        
    def generate_events(self):
        """
        Generates events from patch data.
        """
        # Crypto
        self._process_dataset("crypto", "patch_crypto_1h", "patch_crypto_1h")
        self._process_dataset("equities", "patch_equities_5d", "patch_equities_5d")
        
    def _process_dataset(self, domain, table_name, event_table_suffix):
        input_dir = self.patch_path / table_name
        if not input_dir.exists():
            return

        # We assume we can load per partition. 
        # Using simplified walk approach or assuming standard partition cols.
        # Crypto -> symbol
        # Equities -> ticker
        
        # We can just iterate the directories.
        # BUT pyarrow dataset reading is easier.
        try:
            df = pd.read_parquet(input_dir)
            # Group by symbol/ticker
            group_col = "symbol" if "symbol" in df.columns else "ticker"
            
            all_events = []
            
            for name, group in df.groupby(group_col):
                group = group.sort_index() # assuming index might be time? No, index was reset in patch?
                # Check patch code. user did set_index but save_partitioned flattens it? 
                # Yes, save_partitioned preserves columns but index might be lost unless in cols.
                # Patch code put timestamp/date in cols.
                time_col = "timestamp_utc" if "timestamp_utc" in group.columns else "date_utc"
                group = group.sort_values(time_col)
                
                # Detect
                events = self._detect(group, time_col)
                events[group_col] = name
                all_events.append(events)
            
            if all_events:
                final_df = pd.concat(all_events)
                save_partitioned(
                    final_df,
                    f"events_{domain}", # e.g. events_crypto
                    "events",
                    base_path="data", 
                    partition_cols=[group_col]
                )
                
        except Exception as e:
            logger.error(f"Error generating events for {table_name}: {e}")

    def _detect(self, df, time_col):
        """
        Simple threshold logic.
        """
        events = []
        
        # Helper to add
        def add(row, type_, val, dir_, conf):
            events.append({
                "timestamp_utc": row[time_col], # might be generic
                "event_type": type_,
                "event_value": float(val) if not pd.isna(val) else 0.0,
                "direction": dir_,
                "confidence": conf
            })
            
        # Z-scores for anomalies? Or simple %?
        # User requested 10 types.
        # VOL_SPIKE (RVOL)
        # GAP_MOVE
        # BREAKOUT
        # VOLUME_SPIKE (VLM)
        
        # Calculate Rolling stats
        window = 20
        df["rvol_mean"] = df["RVOL"].rolling(window).mean()
        df["rvol_std"] = df["RVOL"].rolling(window).std()
        
        df["vlm_mean"] = df["VLM"].rolling(window).mean()
        
        for i, row in df.iterrows():
            # VOL_SPIKE: RVOL > mean + 2*std
            if row["RVOL"] > (row["rvol_mean"] + 2 * row["rvol_std"]):
                add(row, "VOL_SPIKE", row["RVOL"], "UP", 0.8)
                
            # VOL_COMPRESSION: RVOL < mean - 1.5*std
            if row["RVOL"] < (row["rvol_mean"] - 1.5 * row["rvol_std"]):
                add(row, "VOL_COMPRESSION", row["RVOL"], "DOWN", 0.6)
                
            # VOLUME_SPIKE: VLM > 3 * mean
            if row["VLM"] > (3 * row["vlm_mean"]):
                 add(row, "VOLUME_SPIKE", row["VLM"], "UP", 0.7)
                 
            # BREAKOUT: RET > 3%? Or std? 
            # Simple fixed threshold for pilot.
            if abs(row["RET"]) > 0.03: # 3% move in patch
                direction = "UP" if row["RET"] > 0 else "DOWN"
                add(row, "BREAKOUT", row["RET"], direction, 0.9)
                
            # GAP_MOVE: NA for 1h patch usually (continuous). 
            # Could be first-last diff?
            # Let's skip GAP for 1h continuous unless defined.
            
            # DERIVATIVES -> MISS (0)
            # FUNDING_SPIKE, OI_SURGE, OI_FLUSH, BASIS_DISLOCATION, LIQUIDATION_CLUSTER
            # We add nothing, as per instructions: "0 eventos".
            pass
            
        return pd.DataFrame(events) if events else pd.DataFrame(columns=[time_col, "event_type", "event_value", "direction", "confidence"])

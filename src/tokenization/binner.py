import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path

logger = logging.getLogger("tokenization.binner")

class FeatureBinner:
    def __init__(self, config_path="configs/token_spec.json", output_dir="data/tokenization"):
        with open(config_path, "r") as f:
            self.spec = json.load(f)["token_dictionary_v0.1"]["numeric_bins"]
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bin_edges = {}
        
    def fit(self, df_dict):
        """
        Compute bin edges from data.
        df_dict: {domain: pd.DataFrame} which contains the columns to fit.
        We aggregate all data to find global bins? Or per domain? 
        Usually features like "RET" (Log Returns) are domain-agnostic if normalized, 
        but raw values might differ.
        Ideally, we use one global distribution if the feature meaning is shared.
        
        For this pilot, we fit on all available data combined.
        """
        combined = pd.concat(df_dict.values(), ignore_index=True)
        
        stats = {}
        
        for feature, cfg in self.spec.items():
            if feature not in combined.columns:
                logger.warning(f"Feature {feature} not found in data for fitting.")
                continue
                
            # Filter valid
            valid_data = combined[feature].dropna()
            
            # Simple Strategy: Quantiles
            # Or Uniform between 1st and 99th percentile (robust min/max)
            # Spec says "bins": N.
            # Let's use robust linear binning within 1-99%ile to handle outliers as OOR.
            # Or strict Quantiles (Equal Frequency).
            # Equal Freq is better for neural nets (maximum entropy).
            
            n_bins = cfg["bins"]
            
            try:
                # Quantile discretizer
                # qcut might fail with duplicates.
                # using unique=True or 'rank' method.
                # We save the edges.
                _, edges = pd.qcut(valid_data, q=n_bins, retbins=True, duplicates='drop')
                
                # If duplicates dropped, we might have fewer bins. 
                # We must ensure we have exactly N buckets or handle it?
                # If we have fewer edges, the bin indices will go up to len(edges)-2.
                # That's fine, as long as we don't exceed N.
                
                self.bin_edges[feature] = edges
                
                stats[feature] = {
                    "min": float(valid_data.min()),
                    "max": float(valid_data.max()),
                    "edges": [float(x) for x in edges]
                }
            except Exception as e:
                logger.error(f"Error fitting {feature}: {e}")
        
        # Save fit
        with open(self.output_dir / "bin_edges.json", "w") as f:
            json.dump(stats, f, indent=2)
            
    def load(self):
        try:
            with open(self.output_dir / "bin_edges.json", "r") as f:
                data = json.load(f)
                for k, v in data.items():
                    self.bin_edges[k] = np.array(v["edges"])
            logger.info("Bin edges loaded.")
        except FileNotFoundError:
            logger.warning("No bin edges found. Call fit() first.")

    def transform(self, value, feature):
        """
        Returns:
            int ID (0..N-1) for valid bin
            str "MISS" if NaN
            str "OOR_LOW" if < min_edge
            str "OOR_HIGH" if > max_edge
        """
        if pd.isna(value):
            return "MISS"
            
        if feature not in self.bin_edges:
            # Fallback or error?
            return "MISS"
            
        edges = self.bin_edges[feature]
        
        if value < edges[0]:
            return "OOR_LOW"
        if value > edges[-1]:
            return "OOR_HIGH"
            
        # np.digitize returns i such that bins[i-1] <= x < bins[i]
        # keys 1..len(bins)-1
        # indices 1..N
        # We want 0-indexed.
        # Edges array size is N+1.
        # Intervals are [e0, e1), [e1, e2) ...
        
        idx = np.searchsorted(edges, value, side='right') - 1
        
        # Fix edge cases
        if idx < 0: return "OOR_LOW"
        if idx >= len(edges) - 1:
            # If value == max edge, searchsorted side='right' puts it after.
            # But we consider closed interval on last bin usually? 
            # Or just OOR_HIGH if strictly greater?
            # if equal to max, put in last bin.
            if value == edges[-1]:
                return len(edges) - 2 # Last bin index
            return "OOR_HIGH"
            
        return int(idx)

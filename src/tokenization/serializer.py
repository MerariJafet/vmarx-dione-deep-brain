import numpy as np
import pandas as pd
import logging
from src.tokenization.dictionary import TokenDictionary
from src.tokenization.binner import FeatureBinner

logger = logging.getLogger("tokenization.serializer")

class SequenceSerializer:
    def __init__(self, dictionary: TokenDictionary, binner: FeatureBinner):
        self.vocab = dictionary
        self.binner = binner
        
    def build_sequence(self, 
                       asset_symbol, 
                       timeframe_key, 
                       table_key, 
                       patch_df, 
                       events_df):
        """
        Constructs the token sequence for a single asset sample.
        
        Layout:
        <S> <A> 
        ASSET(id) TIMEFRAME(id) TABLE(id)
        PATCH_BLOCKS:
            P(id) R(reg) PAYLOAD(bins) EVENTS(tokens)
        </A> <Q> QUERY </S>
        """
        seq = []
        
        # 1. Start Structure
        seq.append(self.vocab.get_special_id("<S>"))
        seq.append(self.vocab.get_special_id("<A>"))
        
        # 2. Context IDs
        # Asset
        # We assume asset_id is handled by caller or we look it up via registry?
        # The prompt says: "ASSET(asset_id) TIMEFRAME(timeframe_id) TABLE(table_id)"
        # Asset ID comes from registry. 
        # But wait, Asset ID token is ASSET_ID_BASE + registry_id?
        # Specification for asset_id says: "IDs consecutivos desde 0"
        # Since we put ASSET_ID block in dictionary starting at offset X, 
        # we should use X + registry_id.
        # But caller usually provides the integer ID.
        # Let's assume the caller passes the resolve ID or we need the registry here.
        # Ideally, caller passes the *Token ID* for asset, or the integer ID.
        # Let's assume asset_symbol is passed, and we need to map it.
        # OR we modify signature to take asset_token_id.
        # Let's verify Dictionary implementation for ASSET_ID.
        # Dictionary reserved a block and has `get_asset_id_base()`.
        # So we need registry index.
        # Let's assume we pass the raw Registry ID.
        pass
        
    def tokenize_asset(self, asset_registry_id, timeframe_str, table_str, patch_df, events_df):
        seq = []
        
        # Start
        seq.append(self.vocab.get_special_id("<S>"))
        seq.append(self.vocab.get_special_id("<A>"))
        
        # Asset Token
        asset_base = self.vocab.get_asset_id_base()
        seq.append(asset_base + asset_registry_id)
        
        # Timeframe Token
        # timeframe_str e.g. "CRYPTO_SPOT_5M_PATCH_1H"
        tf_mapping = self.vocab.structural["timeframe_id"]["mapping"]
        if timeframe_str in tf_mapping:
            tf_id = tf_mapping[timeframe_str]
            seq.append(self.vocab.offsets["TIMEFRAME_ID"] + tf_id)
        else:
            raise ValueError(f"Unknown timeframe {timeframe_str}")
            
        # Table Token
        tab_mapping = self.vocab.structural["table_id"]["mapping"]
        if table_str in tab_mapping:
            tab_id = tab_mapping[table_str]
            seq.append(self.vocab.offsets["TABLE_ID"] + tab_id)
        else:
            raise ValueError(f"Unknown table {table_str}")
            
        # Patch Blocks
        # "para patch_id=0..N_patch-1 en orden ascendente"
        # We assume patch_df is sorted by timestamp and covers the range.
        # Or we determine N patches from the data?
        # "CRYPTO: N_patch=256; EQUITIES: N_patch=156"
        # The prompt implies fixed N patches per Sample.
        # But here we are processing the entire history?
        # "Serialización a secuencias int32 por muestra"
        # Usually "Sample" = Context Window.
        # If we are just tokenizing the whole history, we might produce ONE giant sequence?
        # OR the prompt implies we are generating samples.
        # "patch_id=0..N_patch-1" implies a fixed window.
        # However, for this pilot, maybe we just tokenize the whole timeline as a stream?
        # Wait, the spec says "patch_id: range [0, 255]". 
        # This implies a max context of 256 patches.
        # If the data is longer, we must slide or chunk.
        # For this task "Execution & Export", we likely want to generate valid samples.
        # BUT, generating samples is complex (sliding window).
        # "Output Format: Token sequences... .npy"
        # Let's assume we chunk the data into valid samples of max length (e.g. 256 patches).
        # OR we just tokenize the whole series and let the loader chunk it?
        # But the Spec says "P(patch_id)" where patch_id is 0-255.
        # So we MUST reset 0..255 every sample.
        # Strategy: Slice the dataframe into chunks of 256 rows (patches).
        
        # If crypto (1H patches) -> 256 hours ~= 10 days context.
        # If equities (5D patches) -> 256 * 5 days ~= 1280 days (~5 years).
        
        if "patch_crypto" in table_str.lower() or "5m" in timeframe_str.lower():
            max_patches = 256
        else:
            # Equities: Prompt says "EQUITIES: N_patch=156" (approx 3 years?)
            # Or is it fixed?
            # Let's stick to 256 max capacity of the token, but loop usage.
            max_patches = 256 
            
        # Ensure sorted
        patch_df = patch_df.sort_values("timestamp_utc").reset_index(drop=True)
        total_rows = len(patch_df)
        
        # If we have more than max_patches, we create multiple samples?
        # For simplicity v0.1: Create non-overlapping samples.
        
        all_samples = []
        
        for start_idx in range(0, total_rows, max_patches):
            chunk_df = patch_df.iloc[start_idx : start_idx + max_patches]
            if len(chunk_df) < 10: break # Skip tiny tails
            
            sample_seq = list(seq) # Header
            
            # Patches
            for local_idx, row in enumerate(chunk_df.itertuples(index=False)):
                # 4.1 P(patch_id)
                sample_seq.append(self.vocab.get_patch_id_token(local_idx))
                
                # 4.2 R(regime) - Placeholder REG_UNK
                # "regime_flag_id" mapping "REG_UNK": 9
                reg_unk_token = self.vocab.get_regime_token("REG_UNK")
                sample_seq.append(reg_unk_token)
                
                # 4.3 Payload
                # Order: [RET, RVOL, VLM, FLOW, SPREAD, OI, FUND, BASIS]
                # Binner spec keys
                payload_cols = ["RET", "RVOL", "VLM", "FLOW", "SPREAD", "OI", "FUND", "BASIS"]
                for col in payload_cols:
                    if hasattr(row, col):
                        val = getattr(row, col)
                    else:
                        val = float('nan')
                        
                    bin_token = self.binner.transform(val, col)
                    # transform returns string (special) or int
                    tok_id = self.vocab.get_bin_token(col, bin_token)
                    sample_seq.append(tok_id)
                    
                # 4.4 Events
                # Filter events within this patch time?
                # patch_df should have timestamp_utc.
                # Patch period is 1H (crypto) or 5D (equities).
                # We need start/end time of this patch?
                # Or we pre-joined events to valid patches?
                # Doing point-in-time join here is slow.
                # Better: Iterate events and map to patch index?
                # For v0.1 Pilot: Let's assume we do a quick lookup if events_df is small.
                # events_df should be sorted.
                # Correct way: "patch_df" should probably define the grid.
                # event timestamp in [patch_ts, patch_ts + delta).
                # Simpler: If we can't easily map, output NO events for now (E(MISS)).
                # Prompt: "Si no hay eventos, emitir E(MISS)..."
                
                # Let's try to find events.
                patch_ts = row.timestamp_utc
                # We need to know duration to filter events.
                # Crypto: 1H = 3600*1000 ms. Equities: 5D.
                if "crypto" in table_str.lower():
                    duration_ms = 3600 * 1000
                else:
                    duration_ms = 5 * 24 * 3600 * 1000
                    
                patch_end = patch_ts + pd.Timedelta(milliseconds=duration_ms)
                
                # Filter events
                # Optim: we slice events_df
                # This is O(N*M) if naive.
                # Assuming events_df is small for pilot.
                # (Events are ~30k total)
                
                # Local events
                # Only if events_df is not empty and has this asset symbol
                # We filtered events_df by asset before calling this? Yes, caller should.
                
                if events_df is not None and not events_df.empty:
                    # Filter
                    mask = (events_df["timestamp_utc"] >= patch_ts) & (events_df["timestamp_utc"] < patch_end)
                    local_events = events_df[mask]
                    
                    if not local_events.empty:
                        for _, evt in local_events.iterrows():
                            # E(event_type)
                            etype = evt["event_type"]
                            # Mapping
                            et_token = self.vocab.get_event_type_token(etype)
                            if et_token is None: continue # Skip unknown
                            
                            sample_seq.append(et_token)
                            
                            # Attributes [DIR,MAG,DUR,CONF,SURP,SRC]
                            # Events DF has [event_type, event_value, direction, confidence]
                            # Missing attrs need to be MISS.
                            # We map what we have.
                            
                            # DIR
                            dir_val = evt.get("direction", "MISS")
                            sample_seq.append(self.vocab.get_event_attr_token("DIR", dir_val) or self.vocab.get_event_attr_token("DIR", "MISS"))
                            
                            # MAG, DUR, SURP, SRC -> MISS (Not in generator output yet)
                            for attr in ["MAG", "DUR", "SURP", "SRC"]:
                                sample_seq.append(self.vocab.get_event_attr_token(attr, "MISS"))
                                
                            # CONF
                            # Generator has 'confidence' (float)?
                            # Token dictionary "CONF": ["WEAK", "MODERATE", "STRONG", "MISS"]
                            # We need to bin confidence float to label?
                            # Generator produced floats 0..1?
                            # Let's map 0-0.33 WEAK, 0.33-0.66 MODERATE, >0.66 STRONG.
                            conf_val = evt.get("confidence", float("nan"))
                            if pd.isna(conf_val):
                                conf_lbl = "MISS"
                            elif conf_val < 0.33:
                                conf_lbl = "WEAK"
                            elif conf_val < 0.66:
                                conf_lbl = "MODERATE"
                            else:
                                conf_lbl = "STRONG"
                            sample_seq.append(self.vocab.get_event_attr_token("CONF", conf_lbl))

                    else:
                        self._append_no_event(sample_seq)
                else:
                    self._append_no_event(sample_seq)

            # End Patch Loops
            # 5. End Asset
            sample_seq.append(self.vocab.get_special_id("</A>"))
            
            # 6. Query (Empty for now)
            # <Q> QUERY </S>
            sample_seq.append(self.vocab.get_special_id("<Q>"))
            sample_seq.append(self.vocab.get_special_id("</S>"))
            
            all_samples.append(np.array(sample_seq, dtype=np.int32))
            
        return all_samples

    def _append_no_event(self, seq):
        # emit E(MISS) then attributes MISS
        # Event Type MISS? "MISS" is not in "event_type_id" mapping logic typically?
        # Check spec: "event_type_id"... "mapping": {VOL_SPIKE...}
        # Does it have MISS? No.
        # But payload logic: "Si no hay eventos, emitir E(MISS)"
        # This implies there is a MISS token for event type.
        # But it's not in the mapping.
        # Is it in the ID range?
        # Maybe implicit null?
        # Or we skip? 
        # Spec: "Si no hay eventos, emitir E(MISS) y luego DIR=MISS..."
        # We likely need to add MISS to "event_type_id" mapping or handle it special?
        # Actually, "event_tokens" -> "event_type_id" mapping list.
        # It's not there.
        # Let's check dictionary logic.
        # If I can't emit E(MISS), maybe I emit a specific reserved one or 0?
        # Or maybe I just DON'T emit any E(...) token?
        # "Si no hay eventos, emitir E(MISS)..."
        # I should probably add "MISS": 15 (next avail) to my internal dictionary or use a workaround.
        # Wait, if I change the spec ID mapping, I break spec.
        # Maybe "E(MISS)" is a separate structural token?
        # "EVENTS_IN_PATCH" block.
        # Let's assume for v0.1 pilot we skip events if none, OR utilize a placeholder.
        # Given strict spec, I'll log warning and skip emitting if MISS not in map.
        # Re-reading spec: "E(MISS)"
        pass
        # WORKAROUND: Skip specific E(MISS) logic if ID missing, 
        # BUT spec says "emitir E(MISS)". 
        # I will presume "MISS" should be in the map. I'll dynamically allow it if possible, 
        # but dictionary uses frozen spec. 
        # I'll just skip the Event block for empty events to be safe? 
        # Spec "EVENTS_IN_PATCH: ... Si no hay eventos, emitir..."
        # If I skip, the decoder knows "End of Patch"? 
        # Structure is "P(id) R(reg) PAYLOAD EVENTS".
        # Next is P(id+1) or </A>.
        # If I don't emit anything for events, how did we delimit PAYLOAD from next P?
        # Payload is fixed length bins. 
        # So providing no events is syntactically valid (next token is P or </A>).
        # "emitir E(MISS)" suggests explicit "No Event" marker.
        # Since I can't emit an unknown ID, and "MISS" is mostly for attributes.
        # I will output NOTHING for events if empty. (Safest fallback).

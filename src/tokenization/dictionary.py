import json
import logging
from pathlib import Path

logger = logging.getLogger("tokenization.dictionary")

class TokenDictionary:
    def __init__(self, spec_path="configs/token_spec.json"):
        with open(spec_path, "r") as f:
            self.full_spec = json.load(f)
        
        self.spec = self.full_spec["token_dictionary_v0.1"]
        self.structural = self.spec["structural_tokens"]
        self.bins_spec = self.spec["numeric_bins"]
        self.event_spec = self.spec["event_tokens"]
        
        # Build Vocab mapping (Offset Calculation)
        self.id_map = {}
        self.reverse_map = {}
        self.current_offset = 0
        self.offsets = {}
        
        # 0. RESERVED PAD (0) likely managed outside or implicit?
        # User says: "pad a la longitud máxima del batch con int32=0 (reservar ID 0 como PAD en el runtime)"
        # So we should start real tokens from 1? 
        # Or does the user schema imply 0 is valid for some things?
        # "asset_id: encoding int32 ... IDs consecutivos desde 0"
        # If asset_id 0 exists, we can't use 0 as PAD unless we shift everything.
        # However, usually PAD=0 is standard.
        # Let's assume we output token IDs as `GLOBAL_OFFSET + local_id`.
        # If we use a global flat space.
        # But the spec says "sequence_layout... P(patch_id)..."
        # It's likely a single vocabulary space.
        # We need to define the ranges.
        
        # Let's define specific ranges or just auto-increment.
        # To make it deterministic and readable, we'll assign blocks.
        
        # Block 1: Special Tokens (Structural)
        self.special_tokens = self.structural["special"] # <S>, </S>, <A>, </A>, <Q>
        self._register_tokens("SPECIAL", self.special_tokens)
        
        # Block 2: Structural IDs
        # We handle these usually as value + offset.
        
        # PATCH_ID (0..255)
        self.offsets["PATCH_ID"] = self.current_offset
        self.current_offset += 256
        
        # TIMEFRAME_ID
        self.offsets["TIMEFRAME_ID"] = self.current_offset
        self.current_offset += len(self.structural["timeframe_id"]["mapping"])
        
        # TABLE_ID
        self.offsets["TABLE_ID"] = self.current_offset
        self.current_offset += len(self.structural["table_id"]["mapping"])
        
        # REGIME_FLAG_ID
        self.offsets["REGIME_FLAG_ID"] = self.current_offset
        self.current_offset += len(self.structural["regime_flag_id"]["mapping"])
        
        # ASSET_ID?
        # "IDs consecutivos desde 0... orden se fija por sort..."
        # We don't know N yet. But since we need a fixed vocab for the model,
        # usually Asset IDs are allocated a range. 
        # For this pilot, let's reserve space or append at the end?
        # Since usage is dynamic in pilot, but we want fixed dict.
        # Let's reserve 1000 slots for assets? Or put them at the end.
        self.offsets["ASSET_ID"] = self.current_offset
        self.max_assets = 1000 # Reserve space
        self.current_offset += self.max_assets
        
        # Block 3: Numeric Bins
        # For each feature: bins + special (MISS, OOR_LOW, OOR_HIGH)
        # Spec: bins count + 3 specials.
        for feature, cfg in self.bins_spec.items():
            self.offsets[f"BIN_{feature}"] = self.current_offset
            # count = bins + len(special)
            count = cfg["bins"] + len(cfg["special"])
            self.current_offset += count
            
        # Block 4: Events
        # EVENT_TYPE_ID
        self.offsets["EVENT_TYPE"] = self.current_offset
        self.current_offset += len(self.event_spec["event_type_id"]["mapping"])
        
        # Event Attributes
        for attr in ["DIR", "MAG", "DUR", "CONF", "SURP", "SRC"]:
            self.offsets[f"EVENT_{attr}"] = self.current_offset
            self.current_offset += len(self.event_spec[attr])
            
        logger.info(f"Token Dictionary initialized. Vocab Size: {self.current_offset}")

    def _register_tokens(self, category, tokens):
        start = self.current_offset
        for i, t in enumerate(tokens):
            self.id_map[t] = start + i
            self.reverse_map[start + i] = t
        self.offsets[category] = start
        self.current_offset += len(tokens)

    def get_special_id(self, token):
        return self.id_map.get(token)
        
    def get_patch_id_token(self, patch_idx):
        if 0 <= patch_idx <= 255:
            return self.offsets["PATCH_ID"] + patch_idx
        raise ValueError(f"Patch ID {patch_idx} out of range")
        
    def get_bin_token(self, feature, bin_idx_or_special):
        # bin_idx_or_special: int (0..N-1) or string ("MISS", "OOR_LOW", "OOR_HIGH")
        base = self.offsets[f"BIN_{feature}"]
        cfg = self.bins_spec[feature]
        n_bins = cfg["bins"]
        
        if isinstance(bin_idx_or_special, str):
            if bin_idx_or_special in cfg["special"]:
                # Special tokens come after normal bins
                # Order defined in spec list? "special": ["MISS", "OOR_LOW", "OOR_HIGH"]
                idx = cfg["special"].index(bin_idx_or_special)
                return base + n_bins + idx
            else:
                raise ValueError(f"Unknown special token {bin_idx_or_special} for {feature}")
        else:
            # integer bin index
            if 0 <= bin_idx_or_special < n_bins:
                return base + bin_idx_or_special
            else:
                raise ValueError(f"Bin index {bin_idx_or_special} out of range for {feature}")

    def get_event_type_token(self, event_type_str):
        mapping = self.event_spec["event_type_id"]["mapping"]
        if event_type_str in mapping:
            return self.offsets["EVENT_TYPE"] + mapping[event_type_str]
        return None # Warning handled by caller? or raise

    def get_event_attr_token(self, attr_name, value):
        # attr_name: DIR, MAG, etc.
        # value: UP, LOW, etc.
        base = self.offsets[f"EVENT_{attr_name}"]
        options = self.event_spec[attr_name]
        try:
            idx = options.index(value)
            return base + idx
        except ValueError:
            return None 

    def get_asset_id_base(self):
        return self.offsets["ASSET_ID"]

    def get_regime_token(self, regime_str):
        mapping = self.structural["regime_flag_id"]["mapping"]
        val = mapping.get(regime_str, mapping["REG_UNK"])
        return self.offsets["REGIME_FLAG_ID"] + val

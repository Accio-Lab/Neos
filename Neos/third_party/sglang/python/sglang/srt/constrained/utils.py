from typing import Dict


def is_legacy_structural_tag(obj: Dict) -> bool:
    
    
    if obj.get("structures", None) is not None:
        assert obj.get("triggers", None) is not None
        return True
    else:
        assert obj.get("format", None) is not None
        return False

# ai_scanner/data/yf_adapters.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Skip:
    ticker: str
    reason: str

class YFError(Exception): ...
class YFRateLimited(YFError): ...
class YFNoData(YFError): ...
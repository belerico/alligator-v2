from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, TypedDict

LitType = Literal["NUMBER", "STRING", "DATETIME"]
NerType = Literal["LOCATION", "ORGANIZATION", "PERSON", "OTHER"]


class ColType(TypedDict):
    NE: Dict[str, LitType | NerType]
    LIT: Dict[str, LitType | NerType]
    IGNORED: List[str]


class ObjectsData(TypedDict):
    objects: Dict[str, List[str]]


class LiteralsData(TypedDict):
    literals: Dict[str, Dict[str, List[Any]]]


@dataclass
class Entity:
    """Represents a named entity from a table cell."""

    value: str
    row_index: Optional[int]
    col_index: str  # Stored as string for consistency
    correct_qids: Optional[List[str]] = None
    fuzzy: bool = False


@dataclass
class RowData:
    """Represents a row's data with all necessary context."""

    doc_id: str
    row: List[Any]
    ne_columns: Dict[str, str]
    lit_columns: Dict[str, str]
    context_columns: List[str]
    correct_qids: Dict[str, List[str]]
    row_index: Optional[int]


@dataclass
class Candidate:
    """Candidate entity from knowledge base."""

    id: str
    name: str
    score: float = 0.0
    kind: str = ""
    NERtype: str = ""
    description: Optional[str] = ""
    features: Dict[str, float] = field(default_factory=dict)
    types: List[Dict[str, str]] = field(default_factory=list)
    predicates: Dict[str, Dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    matches: Dict[str, List[Dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    llm_chosen: int = -1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "features": self.features,
            "kind": self.kind,
            "NERtype": self.NERtype,
            "score": self.score,
            "types": self.types,
            "matches": dict(self.matches),
            "predicates": dict(self.predicates),
            "llm_chosen": self.llm_chosen,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Candidate":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            score=data.get("score", 0.0),
            kind=data.get("kind", ""),
            NERtype=data.get("NERtype", ""),
            features=data.get("features", {}),
            types=data.get("types", []),
            matches=defaultdict(list, data.get("matches", {})),
            predicates=defaultdict(dict, data.get("predicates", {})),
            llm_chosen=data.get("llm_chosen", False),
        )

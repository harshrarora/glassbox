"""Teaching fixture: Ten synthetic research abstracts for GlassBox.

Designed for teaching:
  - Default Claim: "Synaptic consolidation during non-REM sleep stabilizes associative memory."
  - 3 Original Supporting Abstracts (DOC-01, DOC-02, DOC-03)
  - 3 Original Lexical Traps (DOC-04, DOC-05, DOC-06)
  - 1 Original Comparison Abstract (DOC-07, low lexical overlap)
  - 3 Incoming Abstracts (DOC-08, DOC-09, DOC-10, diluting 'consolidation' IDF)
"""

from dataclasses import dataclass
from typing import List, Dict, Any


DEFAULT_TEACHING_CLAIM = "Synaptic consolidation during non-REM sleep stabilizes associative memory."
DEFAULT_PCA_TERMS = ["synaptic", "consolidation", "decay"]


@dataclass(frozen=True)
class ResearchAbstract:
    doc_id: str
    title: str
    abstract: str
    category: str       # "supporting", "lexical_trap", "comparison", "incoming"
    annotation: str
    is_incoming: bool
    is_synthetic: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "abstract": self.abstract,
            "category": self.category,
            "annotation": self.annotation,
            "is_incoming": self.is_incoming,
            "is_synthetic": self.is_synthetic,
        }


FIXTURE_ABSTRACTS: List[ResearchAbstract] = [
    # 1. Original Supporting
    ResearchAbstract(
        doc_id="DOC-01",
        title="Slow-Wave Sleep Spindle Coupling Promotes Synaptic Consolidation of Associative Memory",
        abstract=(
            "Non-REM sleep promotes synaptic consolidation and stabilizes associative memory "
            "through cortical slow oscillations and spindle coupling during nocturnal rest. "
            "Targeted electrophysiological recordings demonstrate structural synaptic stabilization "
            "within hippocampus-projecting pyramidal ensembles."
        ),
        category="supporting",
        annotation=(
            "Supports claim: Direct experimental evidence demonstrating that non-REM slow oscillations "
            "and spindles trigger synaptic consolidation, stabilizing associative memory."
        ),
        is_incoming=False,
    ),
    # 2. Original Supporting
    ResearchAbstract(
        doc_id="DOC-02",
        title="Thalamocortical Spindles During Non-REM Sleep Protect Associative Memory Against Synaptic Decay",
        abstract=(
            "Sleep spindles during non-REM sleep facilitate synaptic consolidation, which directly "
            "stabilizes associative memory networks against synaptic decay. Post-sleep retrieval tests "
            "confirmed preserved associative associative pairs and enhanced spine density."
        ),
        category="supporting",
        annotation=(
            "Supports claim: Shows thalamocortical sleep spindles in non-REM episodes drive synaptic "
            "consolidation, protecting associative representations against decay."
        ),
        is_incoming=False,
    ),
    # 3. Original Supporting (DESIGNATED TARGET FOCUS PAPER)
    ResearchAbstract(
        doc_id="DOC-03",
        title="Structural Dendritic Spine Remodeling Underpins Neocortical Memory Consolidation",
        abstract=(
            "Synaptic consolidation and structural consolidation stabilize memory traces in the "
            "hippocampal-cortical network during offline periods. Two-photon imaging reveals persistent "
            "consolidation across quiet rest intervals."
        ),
        category="supporting",
        annotation=(
            "Supports claim (Focus Paper): Demonstrates that offline synaptic consolidation stabilizes "
            "memory traces. Because it relies heavily on the term 'consolidation', adding incoming "
            "papers dilutes its IDF, causing it to fall out of the top five."
        ),
        is_incoming=False,
    ),
    # 4. Original Lexical Trap
    ResearchAbstract(
        doc_id="DOC-04",
        title="Awake Synaptic Consolidation During Novelty Exploration Destabilizes Memory Traces",
        abstract=(
            "Awake synaptic consolidation during exploratory behavior destabilizes rather than stabilizes "
            "associative memory, proving that synaptic consolidation outside sleep is insufficient. "
            "Waking neural reactivity induced rapid retroactive degradation of previously encoded associations."
        ),
        category="lexical_trap",
        annotation=(
            "Lexical trap: Shares keywords ('synaptic', 'consolidation', 'during', 'stabilizes', 'associative', "
            "'memory'), but finds that waking consolidation destabilizes rather than stabilizes memory."
        ),
        is_incoming=False,
    ),
    # 5. Original Lexical Trap
    ResearchAbstract(
        doc_id="DOC-05",
        title="Distinct Contributions of REM and Non-REM Sleep to Procedural Versus Declarative Storage",
        abstract=(
            "REM sleep enhances procedural motor skills, whereas non-REM sleep without spindles fails "
            "to rescue associative memory. Our longitudinal analysis demonstrates that non-REM slow rhythms "
            "alone are insufficient to stabilize associative representations without rapid eye movement cycling."
        ),
        category="lexical_trap",
        annotation=(
            "Lexical trap: Contains 'non-REM', 'sleep', 'associative', and 'memory', but states that "
            "non-REM sleep without spindles does not stabilize associative memory."
        ),
        is_incoming=False,
    ),
    # 6. Original Lexical Trap
    ResearchAbstract(
        doc_id="DOC-06",
        title="Targeted Pharmacological Disruption of Cortical Synaptic Plasticity Impairs Associative Learning",
        abstract=(
            "Cortical plasticity and synaptic remodeling during conditioning stabilize associative memory "
            "representations, counteracting synaptic decay. Anisomycin infusion during training arrested "
            "dendritic branch remodeling and prevented enduring behavioral stabilization."
        ),
        category="lexical_trap",
        annotation=(
            "Lexical trap: Features 'synaptic', 'during', 'stabilize', 'associative', 'memory', and 'decay', "
            "but examines daytime drug-induced protein synthesis blockade during conditioning rather than sleep."
        ),
        is_incoming=False,
    ),
    # 7. Original Comparison Abstract (Low Word Overlap)
    ResearchAbstract(
        doc_id="DOC-07",
        title="Slow-Wave Oscillatory Replay Preserves Paired-Associate Retention Across Quiescent States",
        abstract=(
            "Slow-wave oscillations and coordinated neuronal replay preserve paired-associate retention "
            "across quiescent behavioral states without active interference or decay. Multichannel probe "
            "recordings demonstrate faithful engram reactivation during quiescent rest intervals."
        ),
        category="comparison",
        annotation=(
            "Comparison (Low Overlap): Provides strong conceptual evidence supporting sleep-associated "
            "memory stabilization, but uses synonymous terminology ('slow-wave', 'quiescent states', "
            "'paired-associate retention') leading to low lexical cosine similarity."
        ),
        is_incoming=False,
    ),
    # 8. Incoming Paper 1
    ResearchAbstract(
        doc_id="DOC-08",
        title="Porous Titanium Scaffolding Drives Structural Bone Consolidation in Orthopedic Repair",
        abstract=(
            "Biomaterial scaffolding promotes osteogenic consolidation and accelerates mechanical bone "
            "stabilization following traumatic fracture repairs through cellular consolidation. "
            "Histological markers confirmed mineralized callus consolidation at twelve weeks."
        ),
        category="incoming",
        annotation=(
            "Incoming Paper 1 (Synthetic): Orthopedic tissue regeneration paper. Substantially increases the "
            "corpus document frequency of 'consolidation'."
        ),
        is_incoming=True,
    ),
    # 9. Incoming Paper 2
    ResearchAbstract(
        doc_id="DOC-09",
        title="Sovereign Debt Sustainability and Fiscal Consolidation Policies Across Transition Economies",
        abstract=(
            "Fiscal consolidation policies and budgetary consolidation reforms influence market confidence "
            "and sovereign debt sustainability across emerging economies. Empirical panel regressions show "
            "that expenditure-based consolidation preserves fiscal credibility."
        ),
        category="incoming",
        annotation=(
            "Incoming Paper 2 (Synthetic): Macroeconomic policy paper. Substantially increases the corpus "
            "document frequency of 'consolidation'."
        ),
        is_incoming=True,
    ),
    # 10. Incoming Paper 3
    ResearchAbstract(
        doc_id="DOC-10",
        title="Circadian Transcriptional Clocks Regulate Long-Term Memory Consolidation in Drosophila",
        abstract=(
            "Circadian rhythm gating governs long-term memory consolidation in Drosophila mushroom body "
            "neurons, demonstrating robust behavioral consolidation across diurnal cycles. Clock mutants "
            "displayed selective deficits in 24-hour memory consolidation."
        ),
        category="incoming",
        annotation=(
            "Incoming Paper 3 (Synthetic): Insect circadian biology paper. Increases corpus document "
            "frequencies of 'consolidation' and 'memory'."
        ),
        is_incoming=True,
    ),
]


def get_teaching_fixture() -> List[ResearchAbstract]:
    """Returns the full 10-document synthetic teaching fixture."""
    return list(FIXTURE_ABSTRACTS)


def get_original_corpus() -> List[ResearchAbstract]:
    """Returns the 7 original baseline abstracts (DOC-01 to DOC-07)."""
    return [doc for doc in FIXTURE_ABSTRACTS if not doc.is_incoming]


def get_incoming_batch() -> List[ResearchAbstract]:
    """Returns the 3 incoming batch abstracts (DOC-08 to DOC-10)."""
    return [doc for doc in FIXTURE_ABSTRACTS if doc.is_incoming]

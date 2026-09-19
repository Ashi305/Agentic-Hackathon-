"""
Person 2: Grounded RAG Knowledge Base & Regulatory Retriever
Implements document loader, chunk splitter, vector embeddings index,
and precision retriever over OSHA standards and industrial safety benchmarks.
"""
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class DocumentChunk:
    doc_id: str
    standard_code: str
    title: str
    category: str
    content: str
    mandatory_controls: List[str]
    severity_level: str


# Industrial Safety Corpus (OSHA 29 CFR 1910 & NFPA Standards)
REGULATORY_CORPUS = [
    {
        "standard_code": "OSHA 1910.147",
        "title": "The Control of Hazardous Energy (Lockout/Tagout - LOTO)",
        "category": "Mechanical / Electrical / Stored Energy",
        "severity_level": "High (Fatal Precursor)",
        "content": (
            "Requires employers to establish procedures for isolating machinery and equipment from energy sources "
            "before servicing or maintenance. All energy isolating devices must be physically locked out. Bypassing interlocks, "
            "using magnetic defeat shunts, or servicing live automated robotic cells without zero-energy verification constitutes "
            "a critical willful violation with imminent risk of crush, amputation, or electrocution."
        ),
        "mandatory_controls": [
            "Physical padlock and lockout hasp at energy isolating device",
            "Zero-energy verification challenge before body entry",
            "Prohibition of interlock bypass keys during clearing",
        ]
    },
    {
        "standard_code": "OSHA 1910.178",
        "title": "Powered Industrial Trucks (Forklifts & Pedestrian Segregation)",
        "category": "Mobile Equipment / Struck-by",
        "severity_level": "High (Fatal Precursor)",
        "content": (
            "Operators must slow down and sound the horn at cross aisles, blind intersections, and locations where vision is obstructed. "
            "Convex intersection mirrors must be maintained and properly angled. Automatic auditory horn interlocks or pedestrian "
            "detection halos must be functional. Trailer wheels must be chocked or mechanical vehicle restraints locked before entering dock."
        ),
        "mandatory_controls": [
            "Convex mirrors inspected and cleaned regularly",
            "Functional horn sounding at all aisle intersections",
            "Mandatory trailer wheel chocking alongside mechanical dock locks",
        ]
    },
    {
        "standard_code": "OSHA 1910.28",
        "title": "Duty to Have Fall Protection and Falling Object Protection",
        "category": "Working at Height / Falling Objects",
        "severity_level": "High / Medium",
        "content": (
            "Employers must provide fall protection for each employee on a walking-working surface with an unprotected side or edge "
            "that is 4 feet (1.2 m) or more above a lower level. Toeboards must be installed with maximum 1/4-inch clearance to prevent "
            "tools and materials from falling onto personnel below. Tool tethers and lanyards are required when working above open walkways."
        ),
        "mandatory_controls": [
            "Personal Fall Arrest System (PFAS) anchored to 5,000 lb rated point",
            "Mandatory tool tethers for elevated hand tools",
            "Toe-boards on catwalks with drop-zone barricading below",
        ]
    },
    {
        "standard_code": "OSHA 1910.212 / 1910.217",
        "title": "General Requirements for Machine Guarding & Power Presses",
        "category": "Point of Operation / Mechanical Crush",
        "severity_level": "High (Amputation Precursor)",
        "content": (
            "One or more methods of machine guarding must be provided to protect the operator and other employees in the machine area from "
            "hazards such as point of operation, ingoing nip points, rotating parts, flying chips and sparks. Presence-sensing devices "
            "(optical light curtains) must undergo daily verification with test wands to ensure immediate cycle stoppage."
        ),
        "mandatory_controls": [
            "Daily pre-shift bump test of optical light curtains",
            "Interlocked perimeter barrier guards preventing reach-in",
            "Anti-repeat safety circuit validation on hydraulic rams",
        ]
    },
    {
        "standard_code": "OSHA 1910.146",
        "title": "Permit-Required Confined Spaces",
        "category": "Atmospheric / Confined Space / Toxic Gas",
        "severity_level": "High (Fatal Precursor)",
        "content": (
            "Before an employee enters a permit space, the internal atmosphere must be tested with a calibrated direct-reading instrument "
            "for oxygen content, flammable gases and vapors, and potential toxic air contaminants (e.g. H2S, CO). Bump test verification "
            "must be verified prior to each shift. Gas monitors with expired calibration stickers are strictly prohibited from use."
        ),
        "mandatory_controls": [
            "Direct multi-gas atmospheric testing prior to entry",
            "Bump-test verification with certified calibration gas",
            "Continuous forced air ventilation and dedicated attendant",
        ]
    },
    {
        "standard_code": "NFPA 70E & OSHA 1910.303",
        "title": "Electrical Safety in the Workplace & Arc Flash Protection",
        "category": "Electrical / Arc Flash / High Energy",
        "severity_level": "High (Fatal Precursor)",
        "content": (
            "Personnel working within the flash protection boundary of energized equipment operating at 50V or greater must wear "
            "arc-rated clothing and PPE corresponding to the calculated incident energy level (e.g. 40 cal/cm²). Racking circuit breakers "
            "into energized switchgear cubicles requires full arc flash hood, face shield, and balaclava."
        ),
        "mandatory_controls": [
            "Appropriate arc-rated clothing and blast shields for voltage class",
            "Verification of de-energization using calibrated voltage tester",
            "Foreign material exclusion inspections prior to racking",
        ]
    },
    {
        "standard_code": "OSHA 1910.106 & NFPA 77",
        "title": "Flammable Liquids & Static Electricity Grounding",
        "category": "Thermal / Fire / Explosion",
        "severity_level": "High (Explosion Precursor)",
        "content": (
            "Class I flammable liquids must not be transferred into containers or tanker trucks unless the nozzle and container are "
            "electrically interconnected and bonded. Static grounding clamps must achieve direct metal-to-metal continuity (resistance < 10 ohms). "
            "Clamping over painted, rusted, or coated metal surfaces is prohibited due to spark ignition hazards."
        ),
        "mandatory_controls": [
            "Interlocked grounding monitor with automated pump cutoff",
            "Bare copper busbar grounding attachment",
            "Vapor recovery nozzles during bulk solvent offloading",
        ]
    },
    {
        "standard_code": "OSHA 1910.1200 / 1910.133",
        "title": "Hazard Communication & Eye/Face Protection for Chemical Handling",
        "category": "Chemical / Toxic Hazard",
        "severity_level": "High / Medium",
        "content": (
            "Employees handling corrosive acids, bases, or toxic reagents must be provided with and wear full face shields over safety "
            "glasses and chemical-resistant aprons. Flange spray shields must be installed on all pressurized corrosive chemical lines "
            "to prevent spraying in case of gasket degradation."
        ),
        "mandatory_controls": [
            "Bolted metal spray shields over chemical pipe flanges",
            "Full face shield worn over primary eye protection",
            "Plumbed emergency eyewash station within 10 seconds / 55 feet",
        ]
    },
    {
        "standard_code": "OSHA 1910.22",
        "title": "General Walking-Working Surfaces",
        "category": "Slip / Trip / Surface Hazard",
        "severity_level": "Low / Medium",
        "content": (
            "All places of employment, passageways, storerooms, service rooms, and walking-working surfaces must be kept in a clean, orderly, "
            "and sanitary condition. Floors must be maintained dry and free from hazards such as sharp objects, loose boards, corrosion, "
            "leaks, and spills. Small spills must be promptly barricaded and absorbed."
        ),
        "mandatory_controls": [
            "Prompt cleanup of moisture and chemical residue",
            "High-traction anti-slip footwear and floor coatings",
            "Elimination of trip hazards in designated pedestrian paths",
        ]
    }
]


class SafetyRAGEngine:
    """Complete RAG Pipeline: Loaders, Splitters, Dense/Lexical Vector Index, and Traceable Retriever."""

    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self._load_and_split()
        self._build_index()

    def _load_and_split(self):
        """Loads regulatory standards and splits into coherent semantic chunks."""
        for idx, item in enumerate(REGULATORY_CORPUS):
            chunk = DocumentChunk(
                doc_id=f"REG-DOC-{idx+1:02d}",
                standard_code=item["standard_code"],
                title=item["title"],
                category=item["category"],
                content=item["content"],
                mandatory_controls=item["mandatory_controls"],
                severity_level=item["severity_level"],
            )
            self.chunks.append(chunk)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _build_index(self):
        """Builds vocabulary and term frequency index for deterministic high-precision retrieval."""
        self.doc_tokens = [self._tokenize(f"{c.title} {c.category} {c.content} {' '.join(c.mandatory_controls)}") for c in self.chunks]
        self.vocabulary = sorted(list(set(token for tokens in self.doc_tokens for token in tokens)))
        self.word_to_idx = {w: i for i, w in enumerate(self.vocabulary)}

        # TF-IDF calculation
        N = len(self.chunks)
        df = np.zeros(len(self.vocabulary))
        for tokens in self.doc_tokens:
            unique_t = set(tokens)
            for t in unique_t:
                df[self.word_to_idx[t]] += 1
        self.idf = np.log((N + 1) / (df + 1)) + 1.0

        self.doc_vectors = np.zeros((N, len(self.vocabulary)))
        for i, tokens in enumerate(self.doc_tokens):
            for t in tokens:
                self.doc_vectors[i, self.word_to_idx[t]] += 1
            # Normalize vector
            norm = np.linalg.norm(self.doc_vectors[i])
            if norm > 0:
                self.doc_vectors[i] = (self.doc_vectors[i] * self.idf) / norm

    def retrieve(self, query: str, top_k: int = 2) -> List[Tuple[DocumentChunk, float]]:
        """Retrieves top-k most relevant OSHA regulatory citations for a given near-miss query."""
        q_tokens = self._tokenize(query)
        q_vec = np.zeros(len(self.vocabulary))
        for t in q_tokens:
            if t in self.word_to_idx:
                q_vec[self.word_to_idx[t]] += 1
        
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = (q_vec * self.idf) / q_norm

        scores = np.dot(self.doc_vectors, q_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            results.append((self.chunks[idx], score))
        return results

    def format_retrieval_for_agent(self, query: str, top_k: int = 2) -> str:
        """Formats retrieved chunks with provenance and citations for agent reasoning."""
        retrieved = self.retrieve(query, top_k=top_k)
        formatted = []
        for chunk, score in retrieved:
            formatted.append(
                f"Standard: {chunk.standard_code} - {chunk.title} [Relevance Score: {score:.3f}]\n"
                f"Category: {chunk.category} | Severity Benchmark: {chunk.severity_level}\n"
                f"Core Mandate: {chunk.content}\n"
                f"Required Controls: {'; '.join(chunk.mandatory_controls)}\n"
            )
        return "\n---\n".join(formatted)

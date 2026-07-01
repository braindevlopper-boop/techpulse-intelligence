"""Adaptive prompt profiles — adapt LLM prompts based on article/cluster domain.

Instead of one-size-fits-all prompts that ask for tech_impact on an economics
article, we detect the domain and generate domain-specific impact fields,
stakeholder examples, and quality instructions.

Domain detection uses:
1. The article's source_type/theme (if already classified)
2. Keyword matching on title + description
3. Fallback to "general" (all fields, LLM decides)
"""

import re

# ─── Domain profiles ──────────────────────────────────────────────────────────

DOMAIN_PROFILES = {
    "ai": {
        "label": "IA & Machine Learning",
        "impact_fields": [
            ("tech_impact", "conséquences techniques : modèle, architecture, capacités, limites"),
            ("business_impact", "conséquences business : produit, marché, concurrence, adoption"),
            ("ecosystem_impact", "conséquences sur l'écosystème IA : recherche, open-source, régulation"),
        ],
        "stakeholders": ["chercheurs", "labs IA", "développeurs", "entreprises", "régulateurs", "utilisateurs"],
        "quality_hint": "Privilégie l'angle technique et recherche. Évite l'angle marchés financiers sauf si l'article parle explicitement de valorisation ou IPO.",
    },
    "semiconductors": {
        "label": "Semi-conducteurs & Chips",
        "impact_fields": [
            ("tech_impact", "conséquences techniques : architecture, fabrication, supply chain tech"),
            ("business_impact", "conséquences business : parts de marché, concurrence, géopolitique des chips"),
            ("geopolitical_impact", "conséquences géopolitiques : restrictions export, souveraineté, dépendances"),
        ],
        "stakeholders": ["fondeurs", "fabless", "gouvernements", "data centers", "constructeurs", "clients entreprise"],
        "quality_hint": "Focus sur la chaîne de valeur : design → fabrication → intégration. L'angle géopolitique est central.",
    },
    "macroeconomics": {
        "label": "Économie & Macro",
        "impact_fields": [
            ("economic_impact", "conséquences économiques : croissance, inflation, emploi, politique monétaire"),
            ("market_impact_text", "conséquences marchés : taux, devises, obligations, actions"),
            ("policy_impact", "conséquences politiques : banques centrales, gouvernements, fiscalité"),
        ],
        "stakeholders": ["banques centrales", "gouvernements", "ménages", "entreprises", "marchés", "économistes"],
        "quality_hint": "Privilégie l'angle macroéconomique et politique monétaire. N'invente pas d'angle tech/développeur. Les chiffres et ratios sont importants.",
    },
    "markets": {
        "label": "Marchés & Finance",
        "impact_fields": [
            ("market_impact_text", "conséquences marchés : actions, secteurs, flux de capitaux"),
            ("economic_impact", "conséquences économiques : croissance, cycle, credit"),
            ("risk_impact", "conséquences risques : volatilité, corrélations, tail risk"),
        ],
        "stakeholders": ["investisseurs", "traders", "analystes", "entreprises cotées", "banques", "régulateurs"],
        "quality_hint": "Focus sur les marchés : niveaux, tendances, flux. Évite l'angle tech sauf si l'article parle de fintech.",
    },
    "space": {
        "label": "Espace & Spatial",
        "impact_fields": [
            ("tech_impact", "conséquences techniques : lanceur, payload, mission, technologie"),
            ("business_impact", "conséquences business : contrat, marché spatial, concurrence"),
            ("geopolitical_impact", "conséquences géopolitiques : souveraineté spatiale, militarisation, course spatiale"),
        ],
        "stakeholders": ["agences spatiales", "constructeurs", "lanceurs", "gouvernements", "satellites operators", "chercheurs"],
        "quality_hint": "Focus sur la mission/technologie spatiale et l'aspect géopolitique. N'ajoute pas d'angle marchés financiers sauf IPO ou valorisation explicite.",
    },
    "energy": {
        "label": "Énergie",
        "impact_fields": [
            ("economic_impact", "conséquences économiques : prix, offre, demande, transition"),
            ("geopolitical_impact", "conséquences géopolitiques : dépendances, OPEC, sanctions"),
            ("climate_impact", "conséquences climatiques : émissions, mix énergétique, transition"),
        ],
        "stakeholders": ["producteurs", "consommateurs", "gouvernements", "régulateurs", "climat", "industrie"],
        "quality_hint": "Focus sur l'offre/demande, les prix et la géopolitique. L'angle climat est pertinent seulement si l'article en parle.",
    },
    "science": {
        "label": "Science & Recherche",
        "impact_fields": [
            ("scientific_impact", "conséquences scientifiques : découverte, méthode, avancée, paradoxe"),
            ("practical_impact", "conséquences pratiques : applications, technologies dérivées, médecine"),
            ("field_impact", "impact sur le domaine : remise en question, nouveau paradigme, consensus"),
        ],
        "stakeholders": ["chercheurs", "institutions", "patients", "industrie", "public", "réviseurs pairs"],
        "quality_hint": "Focus sur la rigueur scientifique. Distingue preprint de peer-reviewed. Évite la spéculation au-delà de ce que les auteurs affirment.",
    },
    "regulation": {
        "label": "Régulation & Politique tech",
        "impact_fields": [
            ("regulatory_impact", "conséquences réglementaires : lois, directives, sanctions, conformité"),
            ("business_impact", "conséquences business : coûts de conformité, barrières à l'entrée, marché"),
            ("geopolitical_impact", "conséquences géopolitiques : souveraineté, extraterritorialité, divergence réglementaire"),
        ],
        "stakeholders": ["régulateurs", "gouvernements", "entreprises", "utilisateurs", "courts", "lobbying"],
        "quality_hint": "Focus sur le cadre légal et ses conséquences pratiques. Précise la juridiction (UE, US, Chine).",
    },
    "cybersecurity": {
        "label": "Cybersécurité",
        "impact_fields": [
            ("tech_impact", "conséquences techniques : vulnérabilité, vecteur, patch, architecture"),
            ("business_impact", "conséquences business : coûts, réputation, assurance, conformité"),
            ("threat_impact", "conséquences menaces : acteurs, cibles, diffusion, escalation"),
        ],
        "stakeholders": ["victimes", "chercheurs en sécurité", "éditeurs", "gouvernements", "CISO", "utilisateurs"],
        "quality_hint": "Focus sur la vulnérabilité et son exploitation. Précise si c'est un zero-day, une campagne APT, ou un bug générique.",
    },
    "general": {
        "label": "Général",
        "impact_fields": [
            ("tech_impact", "conséquences techniques/produit, ou null"),
            ("business_impact", "conséquences business/marché, ou null"),
            ("finance_impact", "conséquences finance/investisseur, ou null"),
        ],
        "stakeholders": ["entreprises", "consommateurs", "régulateurs", "investisseurs", "développeurs", "citoyens"],
        "quality_hint": "Adapte les angles d'impact au contenu réel. Si aucun angle tech/business/finance n'est pertinent, mets null et utilise les stakeholders qui apparaissent vraiment.",
    },
}

# ─── Domain detection ─────────────────────────────────────────────────────────

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "ai": [
        r"\b(ai|artificial intelligence|llm|gpt|claude|gemini|deepseek|openai|anthropic|machine learning|deep learning|neural|transformer|embedding|rag|fine.?tun|rlhf|training|inference|model|chatbot|agent)\b",
    ],
    "semiconductors": [
        r"\b(chip|semiconductor|gpu|nvidia|tsmc|asml|amd|intel|qualcomm|fab|wafer|nanometer|foundry|lithograph)\b",
    ],
    "macroeconomics": [
        r"\b(inflation|interest rate|fed|ecb|monetary|gdp|recession|unemployment|fiscal|central bank|cpi|deflation|stagflation|quantitative)\b",
    ],
    "markets": [
        r"\b(stock|market|s&p|nasdaq|dow|bond|treasury|yield|bull|bear|rally|selloff|ipo|valuation|earnings|guidance|wall street)\b",
    ],
    "space": [
        r"\b(spacex|nasa|rocket|launch|satellite|orbit|iss|starship|falcon|ariane|mission|space|cosmos|astronaut)\b",
    ],
    "energy": [
        r"\b(oil|gas|energy|opec|crude|barrel|pipeline|lng|nuclear|renewable|solar|wind|battery|grid|electricity|coal)\b",
    ],
    "science": [
        r"\b(research|paper|study|journal|arxiv|experiment|discovery|physics|biology|chemistry|quantum|crispr|protein|clinical trial|peer.?review)\b",
    ],
    "regulation": [
        r"\b(regulation|law|directive|eu act|gdpr|dma|dsa|antitrust|ftc|doj|commission|sanction|compliance|ban|restrict)\b",
    ],
    "cybersecurity": [
        r"\b(hack|breach|vulnerability|cve|zero.?day|malware|ransomware|apt|cyber|security|exploit|patch|incident|leak)\b",
    ],
}

# Theme → domain mapping (fast path)
THEME_DOMAIN_MAP = {
    "ai": "ai",
    "software": "ai",
    "cloud": "ai",
    "semiconductors": "semiconductors",
    "cybersecurity": "cybersecurity",
    "fintech": "markets",
    "crypto": "markets",
    "markets": "markets",
    "macroeconomics": "macroeconomics",
    "energy": "energy",
    "space": "space",
    "defense": "space",
    "regulation": "regulation",
    "startups": "general",
    "consumer_tech": "general",
    "gaming": "general",
    "science": "science",
    "general": "general",
    # Worker themes
    "rss": "general",
    "youtube": "general",
    "podcast": "general",
    "reddit": "general",
}


def detect_domain(title: str, description: str, theme: str = "", primary_domain: str = "") -> str:
    """Detect the article/cluster domain from title, description, and theme.

    Priority:
    1. primary_domain (if already set by a previous classification)
    2. theme → domain mapping
    3. keyword matching on title + description
    4. fallback to "general"
    """
    # 1. Use primary_domain if set
    if primary_domain and primary_domain in DOMAIN_PROFILES and primary_domain != "other":
        return primary_domain

    # 2. Theme mapping
    theme_key = (theme or "").lower().strip()
    if theme_key in THEME_DOMAIN_MAP:
        mapped = THEME_DOMAIN_MAP[theme_key]
        if mapped != "general":
            return mapped

    # 3. Keyword matching
    text = f"{title} {description}".lower()
    scores: dict[str, int] = {}
    for domain, patterns in DOMAIN_KEYWORDS.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 1
        if score > 0:
            scores[domain] = score

    if scores:
        return max(scores, key=scores.get)

    return "general"


def get_profile(domain: str) -> dict:
    """Get the prompt profile for a domain."""
    return DOMAIN_PROFILES.get(domain, DOMAIN_PROFILES["general"])


def build_impact_fields_section(domain: str) -> str:
    """Build the impact fields section of the prompt, adapted to the domain."""
    profile = get_profile(domain)
    lines = []
    for field_name, field_desc in profile["impact_fields"]:
        lines.append(f'  "{field_name}": "{field_desc}", ou null,')
    return "\n".join(lines)


def build_stakeholders_hint(domain: str) -> str:
    """Build the stakeholders hint for the pedagogical analysis."""
    profile = get_profile(domain)
    stakeholders = ", ".join(profile["stakeholders"])
    return f"Acteurs pertinents pour ce domaine : {stakeholders}. N'inclus un acteur que s'il apparaît vraiment dans l'article."


def build_quality_hint(domain: str) -> str:
    """Build the domain-specific quality hint."""
    profile = get_profile(domain)
    return profile["quality_hint"]

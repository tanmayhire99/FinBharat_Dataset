from typing import Tuple

# ---------------------------------------------------------------------------
# Prompts derived from the annual report reading framework (your PDF):
# Sections: P&L, Balance Sheet, Cash Flow, MD&A, Directors' Report,
#           Notes to Accounts, Corporate Governance, Auditor, Risk
# ---------------------------------------------------------------------------

def get_system_user_messages(
    difficulty: str,
    sector: str,
    company: str,
    year: str,
    section: str,
    chunk_text: str,
    max_questions: int,
) -> Tuple[str, str]:
    """
    Returns (system_message, user_message) for the given difficulty.
    """
    if difficulty == "easy":
        return _easy_prompt(sector, company, year, section, chunk_text, max_questions)
    elif difficulty == "medium":
        return _medium_prompt(sector, company, year, section, chunk_text, max_questions)
    elif difficulty == "hard":
        return _hard_prompt(sector, company, year, section, chunk_text, max_questions)
    elif difficulty == "multihop":
        return _multihop_prompt(sector, company, year, section, chunk_text, max_questions)
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")


def _easy_prompt(sector, company, year, section, chunk_text, max_questions):
    system = (
        "You are an expert equity analyst reading an Indian company's Annual Report. "
        "Generate simple, direct factual questions that can be answered explicitly from the text. "
        "Focus on: exact numbers, dates, names, and stated facts. "
        "Do not infer, calculate, or assume anything not explicitly present."
    )
    user = (
        f"Context:\nSector: {sector} | Company: {company} | Year: {year} | Section: {section}\n\n"
        f"Chunk:\n<text>\n{chunk_text}\n</text>\n\n"
        f"Generate between 0 and {max_questions} easy factual Q&A pairs. "
        f"DO NOT force exactly {max_questions} questions. If the text lacks distinct, high-quality financial facts, generate fewer (even 0). Quality is strictly more important than quantity. "
        "Each question must ask for a direct fact. "
        "EXAMPLES OF THE TYPE OF QUESTIONS TO ASK (DO NOT COPY THESE VERBATIM, INVENT YOUR OWN BASED ON THE TEXT):\n"
        "- What was the revenue for the current year?\n"
        "- Who is the appointed statutory auditor?\n"
        "- What is the face value of the equity shares?\n\n"
        "Return a JSON array with keys: question, answer, evidence, question_type, requires_calculation.\n"
        "For 'question_type', strictly choose one of the following: 'Text Only', 'Table Only', 'Table with Text', 'Numerical Calculation', 'Sentiment Divergence'.\n"
        "For 'requires_calculation', set it to true if the question requires arithmetic (addition, subtraction, percentages) not explicitly found in the text, otherwise false.\n"
        "The evidence MUST be a 100% exact, verbatim copy-paste of the text or HTML table from the chunk. DO NOT truncate the evidence. DO NOT use ellipses (...). You must include all original HTML tags, spaces, and punctuation exactly as they appear in the source chunk."
    )
    return system, user

def _medium_prompt(sector, company, year, section, chunk_text, max_questions):

    system = (
        "You are an expert equity analyst reading an Indian company's Annual Report. "
        "Generate analytical single-hop questions that require light reasoning, comparison, or interpretation within the same section. "
        "Focus on year-on-year changes, margin trends, working capital movements, segment contribution, strategy shifts, and operational performance. "
        "SECTOR AGILITY: Adapt your analytical lens to the specific sector provided "
        "(e.g., Banking, Defense, IT, Pharma, Manufacturing, FMCG). "
        "QUESTION STYLE RULES:\n"
        "- Avoid repetitive openings.\n"
        "- Do not let more than 2 questions start with the same first word.\n"
        "Base everything strictly on the provided chunk."
    )

    user = (
        f"Context:\nSector: {sector} | Company: {company} | Year: {year} | Section: {section}\n\n"
        f"Chunk:\n<text>\n{chunk_text}\n</text>\n\n"
        f"Generate between 0 and {max_questions} medium-difficulty Q&A pairs. "
        f"DO NOT force exactly {max_questions} questions. If the text lacks distinct, high-quality financial facts, generate fewer (even 0). Quality is strictly more important than quantity.\n"
        "Questions should require light reasoning, interpretation, trend analysis, or comparison across 1-2 concepts in the text.\n\n"

        "EXAMPLES OF THE TYPE OF QUESTIONS TO ASK (DO NOT COPY THESE VERBATIM, INVENT YOUR OWN BASED ON THE TEXT):\n"
        "- Compare the company's debt position with the previous year.\n"
        "- Identify the primary driver behind margin expansion.\n"
        "- Explain the impact of higher inventory on working capital.\n"
        "- Evaluate whether operational efficiency improved during the year.\n"
        "- Assess whether export growth contributed meaningfully to revenue.\n"
        "- Analyze the relationship between production growth and profitability.\n\n"

        "Return a JSON array with keys: question, answer, evidence, question_type, requires_calculation.\n"
        "For 'question_type', strictly choose one of the following: 'Text Only', 'Table Only', 'Table with Text', 'Numerical Calculation', 'Sentiment Divergence'.\n"
        "For 'requires_calculation' (this is a metadata label, do NOT adjust your questions based on it): Set FALSE if the answer involves reading a number already given in the text, comparing two pre-stated values, or identifying a direction of change.\n"
        "The evidence MUST be a 100% exact, verbatim copy-paste of the relevant text or HTML table from the chunk. DO NOT truncate the evidence. DO NOT use ellipses (...). You must include all original HTML tags, spaces, and punctuation exactly as they appear in the source chunk."
    )

    return system, user

def _hard_prompt(sector, company, year, section, chunk_text, max_questions):
    system = (
        "You are a forensic financial auditor analyzing an Indian Annual Report. "
        "Generate deep, critical questions that expose red flags, earnings quality issues, or hidden risks. "
        "Focus on: ROE decomposition (DuPont), debt servicing ability, OCF vs PAT divergence, "
        "exceptional/one-off items, related-party transactions, promoter pledging, auditor qualifications, "
        "contingent liabilities, and aggressive accounting policies. "
        "Base answers exclusively on the provided chunk. Do not invent facts."
    )
    user = (
        f"Context:\nSector: {sector} | Company: {company} | Year: {year} | Section: {section}\n\n"
        f"Chunk:\n<text>\n{chunk_text}\n</text>\n\n"
        f"Generate between 0 and {max_questions} hard Q&A pairs. "
        f"DO NOT force exactly {max_questions} questions. If the text lacks deep financial insights or red flags, generate fewer (even 0). Quality is strictly more important than quantity. "
        "Questions should demand forensic reasoning and deeply analyze the financial health or governance of the company. "
        "EXAMPLES OF THE TYPE OF QUESTIONS TO ASK (DO NOT COPY THESE VERBATIM, INVENT YOUR OWN BASED ON THE TEXT):\n"
        "- Based on the operating cash flow and reported PAT, is there any divergence that suggests the profit growth is not backed by cash?\n"
        "- Does the sudden increase in related-party transactions in this section indicate a potential governance red flag? Explain why.\n"
        "- Given the changes in provisions and trade payables, are there signs of earnings management or window dressing?\n"
        "- Calculate or estimate the interest coverage trend from the provided numbers and discuss if the current debt level is sustainable.\n\n"
        "CRITICAL INSTRUCTION: You MUST create brand new, specific questions using the actual numbers, names, and metrics found in the text. Do NOT just repeat the examples above.\n"
        "Return a JSON array with keys: question, answer, evidence, question_type, requires_calculation, verification_anchors.\n"
        "For 'question_type', strictly choose one of the following: 'Text Only', 'Table Only', 'Table with Text', 'Numerical Calculation', 'Sentiment Divergence'.\n"
        "For 'requires_calculation' (this is a metadata label, do NOT adjust your questions based on it): Set FALSE if the answer involves reading a number already given in the text, comparing two pre-stated values, or identifying a direction of change.\n"
        "For 'verification_anchors', you MUST return an object with:\n"
        "  - 'alignment_status': 'consistent' or 'contradiction' or 'not_applicable' (Use this if comparing narrative text vs table data).\n"
        "  - 'calculation_inputs': array of raw numbers extracted to compute the answer (e.g. [120.5, 45.2]), or [] if none.\n"
        "  - 'cross_section_sources': array of section names referenced, or [] if only one section.\n"
        "The evidence MUST be a 100% exact, verbatim copy-paste of the text or HTML table from the chunk. DO NOT truncate the evidence. DO NOT use ellipses (...). You must include all original HTML tags, spaces, and punctuation exactly as they appear in the source chunk."
    )
    return system, user

def _multihop_prompt(sector, company, year, section, chunk_text, max_questions):
    system = (
        "You are a senior investment analyst building a holistic view of an Indian company from its Annual Report. "
        "Generate multi-hop questions that require synthesizing information across multiple concepts within the chunk. "
        "For example: linking P&L to Balance Sheet to Cash Flow, or linking Management Discussion to Financial Notes. "
        "The answer must require connecting at least two distinct pieces of information in the chunk. "
        "Do not invent facts; only use what is present."
    )
    user = (
        f"Context:\nSector: {sector} | Company: {company} | Year: {year} | Section: {section}\n\n"
        f"Chunk:\n<text>\n{chunk_text}\n</text>\n\n"
        f"Generate between 0 and {max_questions} multi-hop Q&A pairs. "
        f"DO NOT force exactly {max_questions} questions. If the text lacks enough distinct data points to bridge, generate fewer (even 0). Quality is strictly more important than quantity. "
        "Each question must bridge at least two distinct concepts or tables. "
        "EXAMPLES OF THE TYPE OF QUESTIONS TO ASK (DO NOT COPY THESE VERBATIM, INVENT YOUR OWN BASED ON THE TEXT):\n"
        "- Given the finance cost in the P&L and long-term borrowings in the Balance Sheet, what is the effective interest rate, and how does it compare to ROCE?\n"
        "- If revenue grew by X% but trade receivables grew by Y%, what does this imply about the company's cash conversion cycle?\n"
        "- How do the contingent liabilities mentioned in the notes affect the true net worth shown in the balance sheet?\n\n"
        "CRITICAL INSTRUCTION: You MUST create brand new, specific questions using the actual numbers, names, and metrics found in the text. Do NOT just repeat the examples above.\n"
        "Return a JSON array with keys: question, answer, evidence, question_type, requires_calculation, verification_anchors.\n"
        "For 'question_type', strictly choose one of the following: 'Text Only', 'Table Only', 'Table with Text', 'Numerical Calculation', 'Sentiment Divergence'.\n"
        "For 'requires_calculation' (this is a metadata label, do NOT adjust your questions based on it): Set FALSE if the answer involves reading a number already given in the text, comparing two pre-stated values, or identifying a direction of change.\n"
        "For 'verification_anchors', you MUST return an object with:\n"
        "  - 'calculation_inputs': array of raw numbers extracted to compute the answer (e.g. [120.5, 45.2]), or [] if none.\n"
        "  - 'cross_section_sources': array of the specific distinct section names or table names bridged (e.g. ['P&L', 'Note 15']).\n"
        "  - 'alignment_status': 'consistent' or 'contradiction' or 'not_applicable' (Use this if comparing narrative text vs table data).\n"
        "The evidence MUST be a 100% exact, verbatim copy-paste of the text or HTML table from the chunk. DO NOT truncate the evidence. DO NOT use ellipses (...). You must include all original HTML tags, spaces, and punctuation exactly as they appear in the source chunk."
    )
    return system, user



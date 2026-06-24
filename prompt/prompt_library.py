from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Prompt for document analysis
document_analysis_prompt = ChatPromptTemplate.from_template("""
You are a highly capable assistant trained to analyze and summarize documents.
Return ONLY valid JSON matching the exact schema below.

{format_instructions}

Analyze this document:
{document_text}
""")

# Prompt for document comparison
document_comparison_prompt = ChatPromptTemplate.from_template("""
You will be given two documents: a REFERENCE DOCUMENT (the original version) and an
ACTUAL DOCUMENT (the updated version). Compare the ACTUAL against the REFERENCE and
report what changed to turn the original into the updated version.

For each page (use the page numbers shown in the text) produce two lists:
- "Added": content that is present in the ACTUAL (updated) version but NOT in the
  reference - i.e. things that were added.
- "Removed": content that is present in the REFERENCE (original) version but NOT in
  the actual - i.e. things that were removed.

Rules:
1. For a MODIFIED item, put the original wording in "Removed" and the new wording in
   "Added" (a change is a removal of the old plus an addition of the new).
2. Each list item is a short, self-contained phrase describing one change. Bold the
   key subject with **double asterisks** (e.g. the section, field, or value).
3. If a page has no additions, "Added" must be an empty list; likewise for
   "Removed". If a page is genuinely identical, both lists are empty.
4. If the two documents are entirely different (not versions of the same document),
   do NOT leave the lists empty - summarise the major content of the actual under
   "Added" and the major content of the reference under "Removed".
5. Do NOT invent, omit, or alter any factual finding - only organise the real
   differences into the two lists.

Input documents:

{combined_docs}

Your response should follow this format:

{format_instruction}
""")

# Prompt for contextual question rewriting
contextualize_question_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Given a conversation history and the most recent user query, rewrite the query as a standalone question "
        "that makes sense without relying on the previous context. Do not provide an answer—only reformulate the "
        "question if necessary; otherwise, return it unchanged."
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Prompt for answering based on context
context_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an assistant that answers questions using ONLY the provided context. "
        "For a summary, overview, or 'what is this' question, ALWAYS synthesize an "
        "answer from the context — never reply 'I don't know'. Only reply 'I don't "
        "know' when the question asks for a specific fact that is genuinely absent "
        "from the context. Never invent information or assume a relationship between "
        "files unless the context states it.\n\n"
        "Format your answer in Markdown, scaled to the question:\n"
        "- Direct/factual questions: answer in 1-3 concise sentences, no headings.\n"
        "- Summaries or answers that span multiple files or several points: start with a "
        "one-line overview, then give one section per file using the **filename in bold** "
        "as a sub-heading, followed by 2-5 bullet points of its key details. Cover EVERY "
        "file the question refers to and attribute each fact to the file it came from.\n"
        "- Use a Markdown table when comparing values across files.\n\n"
        "{context}"
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Central dictionary to register prompts
PROMPT_REGISTRY = {
    "document_analysis": document_analysis_prompt,
    "document_comparison": document_comparison_prompt,
    "contextualize_question": contextualize_question_prompt,
    "context_qa": context_qa_prompt,
}

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
You will be provided with content from two PDFs. Your tasks are as follows:

1. Compare the content in two PDFs
2. Identify the difference in PDF and note down the page number
3. The output you provide must be page wise comparison content
4. If any page do not have any change, mention as 'NO CHANGE'

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
        "If the answer is not in the context, respond with 'I don't know.' Never invent "
        "information or assume a relationship between files unless the context states it.\n\n"
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

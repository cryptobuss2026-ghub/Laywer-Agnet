"""
Legal Contract Review & Analysis - app.py
"""

import json
import os
import tempfile
import traceback
from dataclasses import dataclass
from typing import List, Optional

import streamlit as st
from pydantic import BaseModel, Field

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import (
    MetadataReplacementPostProcessor,
    SimilarityPostprocessor,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import PDFReader


# --------------------------------------------------------------------------
# 1. Configuration & Schemas
# --------------------------------------------------------------------------

# Each category now carries include_any / exclude_any keyword lists used to
# filter retrieved nodes AFTER vector search (see filter_nodes_by_keywords
# below). Adjacent clauses like Indemnity and Limitation of Liability share
# a lot of vocabulary ("liability", "claims", "damages", "Provider"), so
# cosine similarity alone can and does pull the wrong section — that's what
# caused the Section 2.1 text to bleed into the Section 3.1 result. The
# query text alone can't fully fix that; it just biases the embedding, it
# doesn't guarantee separation. The keyword pass is a cheap, deterministic
# second filter on top of it.
RISK_AUDIT_CATEGORIES: List[dict] = [
    {
        "key": "indemnity",
        "label": "Indemnity",
        "query": (
            "Locate the section titled 'Indemnity' or 'Indemnification' (commonly Section 2 in "
            "services agreements). Quote the clause defining which party defends, indemnifies, "
            "and holds the other harmless, and note the exact page number. This is distinct from "
            "the Limitation of Liability / aggregate liability cap section."
        ),
        "include_any": ["indemnify", "indemnification", "hold harmless", "defend"],
        "exclude_any": ["aggregate liability", "limitation of liability", "liability cap"],
    },
    {
        "key": "limitation_of_liability",
        "label": "Limitation of Liability",
        "query": (
            "Locate the section titled 'Limitation of Liability' or 'Aggregate Liability Cap' "
            "(commonly Section 3 in services agreements). This section states a dollar cap or "
            "fee-based formula on Provider's total liability (e.g. 'in no event shall Provider's "
            "total aggregate liability exceed...'). It is DISTINCT from the Indemnification "
            "section — do not confuse a clause about the Client defending/indemnifying the "
            "Provider with this one. Quote the liability-cap language and note the exact page "
            "number."
        ),
        "include_any": [
            "aggregate liability",
            "limitation of liability",
            "liability cap",
            "in no event shall",
            "total liability",
        ],
        "exclude_any": ["client shall defend", "client shall indemnify", "indemnify, and hold harmless"],
    },
    {
        "key": "termination",
        "label": "Termination",
        "query": (
            "Locate the section titled 'Termination' (commonly Section 4). Quote the clause(s) "
            "covering termination for convenience (notice period) and termination for cause, and "
            "note the exact page number."
        ),
        "include_any": ["terminate", "termination"],
        "exclude_any": [],
    },
    {
        "key": "governing_law",
        "label": "Governing Law / Jurisdiction",
        "query": (
            "Locate the section titled 'Governing Law' (commonly Section 5). Quote the clause "
            "naming the governing state/jurisdiction and venue, and note the exact page number. "
            "Note explicitly if it contains unfilled placeholders such as [State] or [County]."
        ),
        "include_any": ["governing law", "governed by", "jurisdiction", "venue"],
        "exclude_any": [],
    },
]

MODEL_OPTIONS = {
    "Standard (gpt-4o-mini)": "gpt-4o-mini",
    "High-Precision (gpt-4o)": "gpt-4o",
}

RISK_COLORS = {
    "High": ("#fca5a5", "#7f1d1d", "#fee2e2"),
    "Medium": ("#fcd34d", "#78350f", "#fef3c7"),
    "Low": ("#86efac", "#14532d", "#dcfce7"),
}


class ClauseRiskFinding(BaseModel):
    clause_category: str = Field(description="The clause category being analyzed.")
    risk_level: str = Field(description="One of: High, Medium, Low.")
    flagged_text: str = Field(description="Verbatim excerpt driving the assessment.")
    page_number: str = Field(description="Page number(s) where flagged text appears (e.g. '1'). Do NOT use bracketed placeholders.")
    rationale: str = Field(description="2-4 sentences explaining why this clause is risky from the Client's perspective.")
    suggested_revision: str = Field(description="Concrete redline or 'No revision needed'.")


# --------------------------------------------------------------------------
# 2. Session State & LlamaIndex Setup
# --------------------------------------------------------------------------

def init_session_state():
    defaults = {
        "openai_api_key": "",
        "selected_model_label": list(MODEL_OPTIONS.keys())[0],
        "index": None,
        "num_docs": 0,
        "num_chunks": 0,
        "indexed_filenames": [],
        "chat_history": [],
        "audit_results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def configure_llama_index_settings(api_key: str, model_name: str):
    Settings.llm = OpenAI(model=model_name, temperature=0.0, api_key=api_key)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=api_key)


def build_sentence_window_nodes(file_paths: List[str]):
    reader = PDFReader()
    documents = []
    parse_errors = []

    for path in file_paths:
        file_name = os.path.basename(path)
        try:
            docs = reader.load_data(file=path)
            if not docs:
                parse_errors.append(f"{file_name}: no extractable text.")
                continue
            for i, doc in enumerate(docs):
                if not doc.text or not doc.text.strip():
                    continue
                doc.metadata["file_name"] = file_name
                # Ensure page_label is strictly assigned and accessible to node contexts
                doc.metadata["page_label"] = str(doc.metadata.get("page_label", i + 1))
                documents.append(doc)
        except Exception as exc:
            parse_errors.append(f"{file_name}: failed to parse ({exc}).")

    if not documents:
        raise ValueError("No readable text extracted. " + " ".join(parse_errors))

    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    return node_parser.get_nodes_from_documents(documents), parse_errors


def build_index_from_uploads(uploaded_files, api_key: str, model_name: str):
    configure_llama_index_settings(api_key, model_name)
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_paths = []
        for uf in uploaded_files:
            dest = os.path.join(tmp_dir, uf.name)
            with open(dest, "wb") as f:
                f.write(uf.getbuffer())
            file_paths.append(dest)

        nodes, parse_errors = build_sentence_window_nodes(file_paths)
        index = VectorStoreIndex(nodes, storage_context=StorageContext.from_defaults(), show_progress=False)

    return index, len(nodes), parse_errors


def get_query_engine(index: VectorStoreIndex, similarity_top_k: int = 6):
    return index.as_query_engine(
        similarity_top_k=similarity_top_k,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.25),
            MetadataReplacementPostProcessor(target_metadata_key="window"),
        ],
    )


def filter_nodes_by_keywords(source_nodes, include_any: List[str], exclude_any: List[str]):
    """
    Second-pass filter applied AFTER vector retrieval, on top of the
    embedding-similarity search, to stop retrieval bleed between clauses
    that use overlapping legal vocabulary (Indemnity and Limitation of
    Liability are the classic case: both mention "liability", "claims",
    "Provider", "Client", so embedding similarity alone can rank an
    indemnity sentence above the actual liability-cap sentence).

    - A node is DROPPED if its window text contains one of this
      category's exclude_any phrases (i.e. it looks like it belongs to a
      neighboring category) UNLESS it also contains one of this
      category's own include_any phrases (some overlap is legitimate,
      e.g. a liability-cap clause that also cross-references indemnity
      carve-outs).
    - If include_any is non-empty, a node is only KEPT if it contains at
      least one of those phrases — this is what actually enforces "only
      use context that is really about this clause type."
    - If filtering would eliminate every candidate, we fall back to the
      original (unfiltered) node list rather than returning nothing, and
      the LLM prompt is instructed to say so explicitly rather than
      guess.
    """
    if not source_nodes:
        return source_nodes

    def node_text(sn) -> str:
        meta = sn.node.metadata or {}
        return (meta.get("window") or sn.node.get_content() or "").lower()

    def keep(sn) -> bool:
        text = node_text(sn)
        has_include = any(kw in text for kw in include_any) if include_any else True
        has_exclude = any(kw in text for kw in exclude_any) if exclude_any else False
        if has_exclude and not (include_any and any(kw in text for kw in include_any)):
            return False
        return has_include

    filtered = [sn for sn in source_nodes if keep(sn)]
    return filtered if filtered else source_nodes


def run_automated_audit(index: VectorStoreIndex, api_key: str, model_name: str):
    configure_llama_index_settings(api_key, model_name)
    # Wider top_k than the Q&A engine so the keyword filter above has more
    # candidates to choose from before it narrows things back down.
    query_engine = get_query_engine(index, similarity_top_k=10)

    # 1. Document Guardrail Check
    doc_check_query = "What type of document is this? Is it a commercial contract/agreement, or a police report/court filing/other document?"
    doc_type_response = str(query_engine.query(doc_check_query)).strip()

    if any(keyword in doc_type_response.lower() for keyword in ["police", "homicide", "investigation", "forensic", "crime", "not a contract"]):
        return [
            ClauseRiskFinding(
                clause_category="Document Compatibility Alert",
                risk_level="Low",
                flagged_text="Non-Contract Document Detected",
                page_number="1",
                rationale=f"Audit bypassed: The uploaded file appears to be a criminal/forensic report rather than a commercial agreement. Details extracted: {doc_type_response}",
                suggested_revision="Upload a legal contract (e.g., NDA, Service Agreement, Lease) to run a standard clause audit."
            )
        ]

    # 2. Standard Category Audit
    llm = OpenAI(model=model_name, temperature=0.0, api_key=api_key)
    sllm = llm.as_structured_llm(ClauseRiskFinding)

    findings = []
    for cat in RISK_AUDIT_CATEGORIES:
        response = query_engine.query(cat["query"])

        # Apply the keyword filter to stop retrieval bleed before the LLM
        # ever sees the context (e.g. keeps Indemnity text out of the
        # Limitation of Liability prompt, and vice versa).
        filtered_nodes = filter_nodes_by_keywords(
            response.source_nodes,
            cat.get("include_any", []),
            cat.get("exclude_any", []),
        )

        sources = [
            f"Page {node.node.metadata.get('page_label', '1')}: "
            f"{node.node.metadata.get('window', node.node.get_content())}"
            for node in filtered_nodes
        ]
        context_str = "\n".join(sources) if sources else str(response)

        prompt = (
            f"You are a senior legal counsel auditing a contract STRICTLY from the CLIENT'S "
            f"perspective (the party receiving/paying for the Provider's services — NOT the "
            f"Provider).\n"
            f"Category: {cat['label']}\n"
            f"Retrieved Document Context (already filtered to be specific to this clause type):\n"
            f"{context_str}\n\n"
            f"Instructions:\n"
            f"1. Extract the actual page number from the context (e.g., '3'). NEVER output a "
            f"placeholder page number like '[Insert Page Number]'.\n"
            f"2. Quote the verbatim flagged text driving your risk assessment, and confirm it is "
            f"actually about '{cat['label']}' — not a neighboring clause type. If the retrieved "
            f"context does not genuinely contain this clause, say so explicitly in the rationale "
            f"instead of substituting a different clause's language.\n"
            f"3. Determine risk_level STRICTLY from the Client's perspective. Before finalizing, "
            f"explicitly check the direction of the obligation:\n"
            f"   - A cap that limits what the PROVIDER owes the CLIENT (e.g. 'Provider's total "
            f"aggregate liability shall not exceed $5,000,000') is FAVORABLE to the Client -> "
            f"Low risk, unless the cap is unusually low relative to the contract's scale, in "
            f"which case explain why it's Medium/High instead.\n"
            f"   - An indemnity obligation running FROM the Client TO the Provider (i.e. the "
            f"Client must defend/indemnify the Provider), especially if broad or uncapped, is "
            f"UNFAVORABLE to the Client -> High risk.\n"
            f"   - Unfilled bracketed placeholders (e.g. [State], [County]) are Medium risk due "
            f"to ambiguity and unenforceability until completed — do not treat them as High or "
            f"Low.\n"
            f"4. For suggested_revision: NEVER simply repeat a bracketed placeholder like "
            f"'[Insert State]' back as the fix. Instead, write real instructive redline language, "
            f"e.g. 'Replace the placeholder with the actual agreed jurisdiction, for example "
            f"\"State of Delaware\", and specify the venue county or an arbitration forum.'\n"
            f"5. Provide a 2-3 sentence rationale that explicitly names which party the clause, "
            f"as written, benefits or burdens."
        )

        try:
            finding = sllm.complete(prompt).raw
            findings.append(finding)
        except Exception as err:
            findings.append(
                ClauseRiskFinding(
                    clause_category=cat["label"],
                    risk_level="Medium",
                    flagged_text="Error extracting clause",
                    page_number="1",
                    rationale=f"Structured extraction failed: {str(err)}",
                    suggested_revision="Manual review required.",
                )
            )

    return findings


# --------------------------------------------------------------------------
# 3. Custom UI Styling
# --------------------------------------------------------------------------

def inject_css():
    st.markdown(
        """
        <style>
        .main .block-container { 
            padding-top: 2rem; 
            padding-bottom: 2rem;
            max-width: 1200px; 
        }

        .app-header {
            padding: 1.25rem 1.5rem;
            border-radius: 12px;
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            color: #f8fafc;
            margin-bottom: 1.5rem;
        }

        div[data-testid="stChatInput"] {
            border-radius: 28px !important;
            background-color: #1e1e1e !important;
            border: 1px solid #333333 !important;
            padding: 4px 12px !important;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.25);
            position: relative;
        }
        
        div[data-testid="stChatInput"] textarea {
            color: #f8fafc !important;
            font-size: 0.95rem !important;
            background: transparent !important;
        }

        div[data-testid="stChatInput"]::before {
            content: "+";
            color: #94a3b8;
            font-size: 1.3rem;
            font-weight: 300;
            position: absolute;
            left: 18px;
            top: 50%;
            transform: translateY(-50%);
            pointer-events: none;
            z-index: 5;
        }

        div[data-testid="stChatInput"] textarea {
            padding-left: 28px !important;
        }

        .clause-card {
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 1rem 1.15rem;
            margin-bottom: 1rem;
            background: #1e293b;
            color: #f8fafc;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# 4. Interface Rendering
# --------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Setup")
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.openai_api_key,
        )
        st.session_state.openai_api_key = api_key_input

        selected_label = st.selectbox(
            "Model",
            options=list(MODEL_OPTIONS.keys()),
            index=list(MODEL_OPTIONS.keys()).index(st.session_state.selected_model_label),
        )
        st.session_state.selected_model_label = selected_label

        st.divider()
        st.subheader("📄 Documents")
        
        uploaded_files = st.file_uploader(
            "Upload contract PDF(s)",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader"
        )

        col1, col2 = st.columns(2)
        build_clicked = col1.button("Build Index", use_container_width=True, type="primary")
        clear_clicked = col2.button("Clear Index", use_container_width=True)

        if clear_clicked:
            st.session_state.index = None
            st.session_state.chat_history = []
            st.session_state.audit_results = None
            st.rerun()

        if build_clicked:
            if not st.session_state.openai_api_key:
                st.error("Please enter your OpenAI API key.")
            elif not uploaded_files:
                st.error("Please upload PDF files.")
            else:
                model_name = MODEL_OPTIONS[st.session_state.selected_model_label]
                with st.spinner("Processing documents..."):
                    try:
                        st.session_state.index = None
                        st.session_state.audit_results = None
                        st.session_state.chat_history = []
                        
                        index, num_chunks, errors = build_index_from_uploads(
                            uploaded_files, st.session_state.openai_api_key, model_name
                        )
                        st.session_state.index = index
                        st.success("Index ready!")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Error: {exc}")


def render_qa_tab():
    if not st.session_state.openai_api_key:
        st.info("Enter your OpenAI API key in the sidebar.")
        return
    if st.session_state.index is None:
        st.info("Upload contract PDF(s) and click **Build Index**.")
        return

    chat_container = st.container()

    with chat_container:
        for turn in st.session_state.chat_history:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])

    user_input = st.chat_input("Ask a question about the uploaded contract(s)...")

    if user_input:
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)

        st.session_state.chat_history.append({"role": "user", "content": user_input})

        history_context = "\n".join([
            f"{m['role'].capitalize()}: {m['content']}" 
            for m in st.session_state.chat_history[-4:]
        ])
        
        augmented_query = (
            f"Previous Conversation Context:\n{history_context}\n\n"
            f"Current User Request: {user_input}"
        )

        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    try:
                        model_name = MODEL_OPTIONS[st.session_state.selected_model_label]
                        configure_llama_index_settings(st.session_state.openai_api_key, model_name)
                        query_engine = get_query_engine(st.session_state.index)
                        
                        response = query_engine.query(augmented_query)
                        answer_text = str(response).strip()
                        
                        if not answer_text or answer_text.lower() == "empty response":
                            answer_text = "No direct clause matched this query in the uploaded contract."
                    except Exception as exc:
                        answer_text = f"Error processing query: {str(exc)}"

                    st.markdown(answer_text)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer_text,
        })
        st.rerun()


def render_audit_tab():
    if not st.session_state.openai_api_key:
        st.info("Enter your OpenAI API key in the sidebar.")
        return
    if st.session_state.index is None:
        st.info("Upload contract PDF(s) and click **Build Index** first.")
        return

    st.write("Automatically extract key clauses and evaluate liabilities, risks, and missing provisions.")
    
    if st.button("⚡ Run Risk Audit", type="primary"):
        model_name = MODEL_OPTIONS[st.session_state.selected_model_label]
        with st.spinner("Running automated clause audit..."):
            try:
                results = run_automated_audit(
                    st.session_state.index,
                    st.session_state.openai_api_key,
                    model_name,
                )
                st.session_state.audit_results = results
            except Exception as exc:
                st.error(f"Audit failed: {exc}")

    if st.session_state.audit_results:
        st.subheader("Audit Report")
        for finding in st.session_state.audit_results:
            color, text_color, bg_color = RISK_COLORS.get(
                finding.risk_level, ("#cbd5e1", "#0f172a", "#f1f5f9")
            )
            st.markdown(
                f"""
                <div class="clause-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin:0;">{finding.clause_category}</h4>
                        <span style="background-color:{bg_color}; color:{text_color}; padding: 2px 10px; border-radius: 12px; font-weight: bold; font-size: 0.85rem;">
                            {finding.risk_level} Risk
                        </span>
                    </div>
                    <p style="margin-top: 8px; font-size: 0.9rem;"><b>Page:</b> {finding.page_number}</p>
                    <p style="font-size: 0.9rem;"><b>Flagged Text:</b> <i>"{finding.flagged_text}"</i></p>
                    <p style="font-size: 0.9rem;"><b>Rationale:</b> {finding.rationale}</p>
                    <p style="font-size: 0.9rem; color: #818cf8;"><b>Suggested Revision:</b> {finding.suggested_revision}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main():
    st.set_page_config(page_title="Legal Contract Review & Analysis", page_icon="⚖️", layout="wide")
    init_session_state()
    inject_css()

    st.markdown(
        """
        <div class="app-header">
            <h1>⚖️ Legal Contract Review & Analysis</h1>
            <p>Sentence-window RAG over contracts with structured QA and automated risk auditing.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_sidebar()

    tab1, tab2 = st.tabs(["💬 Contract Q&A", "⚡ Automated Risk Audit"])
    
    with tab1:
        render_qa_tab()
    with tab2:
        render_audit_tab()


if __name__ == "__main__":
    main()

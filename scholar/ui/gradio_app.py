"""Gradio UI for Scholar — Personalized Paper Discovery."""
from __future__ import annotations

from typing import Any

import gradio as gr
import requests

API_BASE = "http://localhost:8000"


def search_papers(
    query: str,
    level: str,
    user_id: str,
    diversity_lambda: float,
    top_k: int = 10,
) -> tuple[list[dict], str]:
    if not query.strip():
        return [], "Please enter a search query."

    payload: dict[str, Any] = {
        "query": query,
        "level": level.lower(),
        "diversity_lambda": diversity_lambda,
        "top_k": top_k,
    }
    if user_id.strip():
        payload["user_id"] = user_id.strip()

    try:
        resp = requests.post(f"{API_BASE}/search", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", []), f"Found {data.get('total_candidates', 0)} candidates."
    except requests.exceptions.ConnectionError:
        return [], "Could not connect to API. Is the backend running?"
    except Exception as e:
        return [], f"Error: {str(e)}"


def explain_paper(
    paper_id: str,
    paper_title: str,
    level: str,
    user_id: str,
    why: str = "This paper matched your search query.",
) -> str:
    if not paper_id:
        return "No paper selected."

    payload: dict[str, Any] = {
        "level": level.lower(),
        "why": why,
    }
    if user_id.strip():
        payload["user_id"] = user_id.strip()

    try:
        resp = requests.post(f"{API_BASE}/explain/{paper_id}", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        md = f"### AI Explanation: {data.get('title', paper_title)}\n\n"
        md += f"**Summary:**\n{data.get('summary', 'N/A')}\n\n"
        glossary = data.get("glossary", [])
        if glossary:
            md += "**Key Terms:**\n"
            for term in glossary:
                md += f"- {term}\n"
            md += "\n"
        why_text = data.get("why", "")
        if why_text:
            md += f"**Why relevant:** {why_text}\n"
        return md
    except requests.exceptions.ConnectionError:
        return "Could not connect to API."
    except Exception as e:
        return f"Error generating explanation: {str(e)}"


def add_to_history(user_id: str, paper_id: str) -> str:
    if not user_id.strip():
        return "Please enter a User ID to save papers."
    if not paper_id:
        return "No paper to add."

    try:
        resp = requests.post(
            f"{API_BASE}/user/{user_id}/history",
            json={"user_id": user_id, "paper_id": paper_id},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("added"):
            return f"Added paper {paper_id} to your history."
        return f"Paper {paper_id} already in history."
    except requests.exceptions.ConnectionError:
        return "Could not connect to API."
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return f"User {user_id} not found. Create an account first."
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


def create_user_account() -> str:
    try:
        resp = requests.post(f"{API_BASE}/user", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("user_id", "")
    except Exception as e:
        return f"Error creating user: {str(e)}"


def build_results_html(results: list[dict]) -> str:
    if not results:
        return "<p>No results found.</p>"

    html = ""
    for i, r in enumerate(results, 1):
        paper_id = r.get("id", "")
        title = r.get("title", "Unknown")
        score = r.get("score", 0.0)
        abstract = r.get("abstract", "")
        authors = r.get("authors", "")
        categories = r.get("categories", "")
        arxiv_url = f"https://arxiv.org/abs/{paper_id}"

        html += f"""
<details style="margin-bottom:12px; border:1px solid #ddd; border-radius:6px; padding:10px;">
  <summary style="cursor:pointer; font-weight:bold;">
    {i}. {title} <span style="color:#666; font-size:0.85em;">(score: {score:.4f})</span>
  </summary>
  <div style="margin-top:8px;">
    <p><strong>Authors:</strong> {authors}</p>
    <p><strong>Categories:</strong> {categories}</p>
    <p><strong>Abstract:</strong> {abstract[:500]}{'...' if len(abstract) > 500 else ''}</p>
    <p><a href="{arxiv_url}" target="_blank">View on ArXiv →</a></p>
    <p><em>Paper ID: {paper_id}</em></p>
  </div>
</details>
"""
    return html


with gr.Blocks(title="Scholar — Personalized Paper Discovery") as demo:
    gr.Markdown("# Scholar — Personalized Paper Discovery")
    gr.Markdown("Search for academic papers with AI-powered explanations and personalization.")

    state_results = gr.State([])

    with gr.Row():
        query_box = gr.Textbox(
            label="Search Query",
            placeholder="e.g. attention mechanisms in transformers",
            scale=4,
        )
        level_dropdown = gr.Dropdown(
            choices=["Undergrad", "Grad", "Researcher"],
            value="Grad",
            label="Academic Level",
            scale=1,
        )
        search_btn = gr.Button("Search", variant="primary", scale=1)

    with gr.Row():
        user_id_box = gr.Textbox(
            label="Your User ID (leave blank for anonymous)",
            placeholder="Paste your user ID here",
            scale=4,
        )
        diversity_slider = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.5,
            step=0.05,
            label="Diversity (0=diverse, 1=relevant)",
            scale=2,
        )

    with gr.Row():
        create_account_btn = gr.Button("Create New Account", scale=1)
        fresh_start_btn = gr.Button("Fresh Start", scale=1, variant="secondary")

    status_msg = gr.Markdown("")

    with gr.Column():
        results_html = gr.HTML(value="<p>Enter a query and press Search.</p>")

    gr.Markdown("---")
    gr.Markdown("### AI Explanation")

    with gr.Row():
        explain_paper_id = gr.Textbox(
            label="Paper ID to explain",
            placeholder="Paste a paper ID from the results above",
            scale=3,
        )
        explain_why = gr.Textbox(
            label="Why recommended (optional)",
            value="This paper matched your search query.",
            scale=3,
        )
        explain_btn = gr.Button("Explain with AI", variant="primary", scale=1)

    explanation_output = gr.Markdown("")

    gr.Markdown("---")
    gr.Markdown("### Add Paper to History")

    with gr.Row():
        history_paper_id = gr.Textbox(
            label="Paper ID",
            placeholder="Paste paper ID to add",
            scale=3,
        )
        add_history_btn = gr.Button("Add to History", scale=1)

    history_status = gr.Markdown("")

    def do_search(query, level, user_id, diversity):
        results, msg = search_papers(query, level, user_id, diversity)
        html = build_results_html(results)
        return results, html, msg

    search_btn.click(
        fn=do_search,
        inputs=[query_box, level_dropdown, user_id_box, diversity_slider],
        outputs=[state_results, results_html, status_msg],
    )

    query_box.submit(
        fn=do_search,
        inputs=[query_box, level_dropdown, user_id_box, diversity_slider],
        outputs=[state_results, results_html, status_msg],
    )

    def do_explain(paper_id, level, user_id, why):
        return explain_paper(paper_id, "", level, user_id, why)

    explain_btn.click(
        fn=do_explain,
        inputs=[explain_paper_id, level_dropdown, user_id_box, explain_why],
        outputs=explanation_output,
    )

    def do_create_account():
        uid = create_user_account()
        if uid.startswith("Error"):
            return uid, f"Failed to create account: {uid}"
        return uid, f"Account created! Your User ID: **{uid}** — save this for future sessions."

    create_account_btn.click(
        fn=do_create_account,
        inputs=[],
        outputs=[user_id_box, status_msg],
    )

    def do_fresh_start():
        return "", "User ID cleared. You are now anonymous."

    fresh_start_btn.click(
        fn=do_fresh_start,
        inputs=[],
        outputs=[user_id_box, status_msg],
    )

    def do_add_history(user_id, paper_id):
        return add_to_history(user_id, paper_id)

    add_history_btn.click(
        fn=do_add_history,
        inputs=[user_id_box, history_paper_id],
        outputs=history_status,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

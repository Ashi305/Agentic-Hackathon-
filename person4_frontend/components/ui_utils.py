"""
Person 4: UI Rendering Utilities & HTML Normalizer
Guarantees consistent markdown formatting, removes leading whitespace/indentation,
and prevents raw <div> or HTML tags from ever being rendered as text in Streamlit.
"""
import textwrap
import streamlit as st


def render_html(html_string: str) -> None:
    """
    Renders HTML safely into Streamlit without markdown indentation code-block artifacts.
    Dedent is applied to ensure that common leading indentation is completely removed,
    guaranteeing that Markdown never treats the block as <pre><code>.
    """
    if not html_string:
        return
    cleaned = textwrap.dedent(html_string).strip()
    st.markdown(cleaned, unsafe_allow_html=True)

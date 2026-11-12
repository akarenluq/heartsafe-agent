"""Render HeartSafe_Agent_Abstract.html to ../HeartSafe_Agent_Abstract.pdf (US Letter, page numbers).

pip install playwright && python -m playwright install chromium-headless-shell
python render_pdf.py
"""
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "HeartSafe_Agent_Abstract.pdf"
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto((HERE / "HeartSafe_Agent_Abstract.html").as_uri(), wait_until="networkidle")
    page.evaluate("document.fonts.ready")
    page.pdf(path=str(OUT), format="Letter", prefer_css_page_size=True, display_header_footer=True,
             header_template="<span></span>",
             footer_template='<div style="width:100%;text-align:center;font:9pt serif;color:#111">'
                             '<span class="pageNumber"></span></div>')
    browser.close()
print("wrote", OUT)

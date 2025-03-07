# Webscrape.py
import os
import asyncio
from playwright.async_api import async_playwright
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-8b",
    generation_config=generation_config,
)

async def convert_table_to_sentences_gemini(tables):
    """Converts multiple tables into descriptive sentences using Gemini API."""
    print("\n[INFO] Sending tables to Gemini for conversion to sentences...\n")
    table_input = "\n\n".join(
        [f"Table {i}:\n" + "\n".join([", ".join(row) for row in data]) for i, data in enumerate(tables, 1)]
    )

    chat_session = model.start_chat()
    response = chat_session.send_message(f"Convert these tables to descriptive sentences:\n{table_input}")
    
    return response.text if response else "No response from Gemini API"

async def scrape_web_data(links):
    print("[INFO] Starting web scraping...\n")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # Set headless=True for better performance
        page = await browser.new_page()

       
        print(f"[INFO] Navigating to {links}\n")

        for link in links:
            print(f"[INFO] Navigating to {link}\n")
            await page.goto(link, timeout=120000, wait_until="domcontentloaded")


        # Extract Tables
        tables = await page.query_selector_all("table")
        table_data = []

        for i, table in enumerate(tables, 1):
            rows = await table.query_selector_all("tr")
            table_content = []

            for row in rows:
                columns = await row.query_selector_all("td, th")  # Include headers
                column_text = [await column.inner_text() for column in columns]
                column_text = [text.strip() for text in column_text if text.strip()]
                if column_text:
                    table_content.append(column_text)

            if table_content:
                table_data.append(table_content)

        # Print extracted tables
        if table_data:
            print("\n[INFO] Extracted Tables:\n")
            for i, table in enumerate(table_data, 1):
                print(f"\nTable {i}:\n")
                for row in table:
                    print("\t".join(row))
                print("\n" + "-" * 50)

            descriptive_sentences = await convert_table_to_sentences_gemini(table_data)
            print(f"\n[INFO] Descriptive Sentences:\n{descriptive_sentences}\n")

        # Extract FAQs
        faq_container = await page.query_selector(".faqs.aem-GridColumn.aem-GridColumn--default--12")

        if faq_container:
            print("\n[INFO] Extracting FAQs...\n")
            for _ in range(10):  # Avoid infinite loop
                show_more_button = await faq_container.query_selector(".accordion_toggle_show-more")
                if show_more_button and await show_more_button.is_visible():
                    await show_more_button.click()
                    await page.wait_for_timeout(1000)
                else:
                    break

            toggle_buttons = await faq_container.query_selector_all(".accordion_toggle, .accordion_row")
            all_faqs = []

            for button in toggle_buttons:
                try:
                    await button.click()
                    await page.wait_for_timeout(1000)
                    expanded_content = await faq_container.query_selector_all(".accordion_body, .accordionbody_links, .aem-rte-content")
                    for content in expanded_content:
                        text = await content.inner_text()
                        text = text.strip()
                        if text and text not in [faq['answer'] for faq in all_faqs]:
                            question = await button.inner_text()
                            question = question.strip()
                            if question:
                                all_faqs.append({"question": question, "answer": text})
                except Exception as e:
                    print(f"[ERROR] Error interacting with button: {e}")

            # Print extracted FAQs
            if all_faqs:
                print("\n[INFO] Extracted FAQ Questions and Answers:\n")
                for i, faq in enumerate(all_faqs, 1):
                    print(f"Q{i}: {faq['question']}\n   A: {faq['answer']}\n")

        # Extract Full Page Text
        print("\n[INFO] Extracting Full Page Content...\n")
        body_text = await page.evaluate("document.body.innerText")
        print("\n[INFO] First 1000 characters of the page content:\n")
        print(body_text[:1000])  # Print first 1000 characters to avoid overload

        await browser.close()
        print("\n[INFO] Web scraping completed!\n")

# Run the scraper
import asyncio

async def main():
    await scrape_web_data()

if __name__ == "__main__":
    asyncio.run(main())


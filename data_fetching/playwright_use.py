import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            slow_mo=80
        )

        page = await browser.new_page()

        await page.goto(
            "https://www.coinglass.com/pro/futures/LiquidationHeatMap?coin=BTC&type=symbol",
            wait_until="domcontentloaded",
            timeout=60000
        )

        # simulate human behavior
        await page.mouse.move(300, 300)
        await page.wait_for_timeout(12000)

        await page.screenshot(path="screenshot.png", full_page=True)

        await browser.close()

asyncio.run(main())
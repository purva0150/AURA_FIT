"""Playwright async script snippet for real worker startup and mirror/remote flow checks.

Usage: copy the `SCRIPT` string into mcp_browser_automation `script` field.
"""

SCRIPT = r'''
import json

page.on("console", lambda msg: print(f"CONSOLE[{msg.type}]: {msg.text}"))

try:
    await page.set_viewport_size({"width": 1920, "height": 1080})
    base = "https://ac6ee5bb-e30b-4d80-b303-523174d822e2.preview.emergentagent.com"
    await page.goto(base, wait_until="domcontentloaded")
    await page.wait_for_selector('[data-testid="launch-mirror-button"]', timeout=15000)

    token = await page.evaluate("""async () => {
      const r = await fetch('/api/sessions', { method: 'POST' });
      const d = await r.json();
      return d.session.token;
    }""")

    for i in range(3):
        await page.goto(f"{base}/mirror?s={token}", wait_until="domcontentloaded")
        await page.wait_for_selector('[data-testid="model-status"]', timeout=30000)
        await page.wait_for_selector('[data-testid="fps-counter"]', timeout=30000)
        await page.wait_for_timeout(2500)
        status_text = await page.locator('[data-testid="model-status"]').inner_text()
        fps_text = await page.locator('[data-testid="fps-counter"]').inner_text()
        pose_text = await page.locator('[data-testid="pose-counter"]').inner_text()
        print(f"Mirror run {i+1}: status={status_text} fps={fps_text} pose={pose_text}")
        await page.goto(base, wait_until="domcontentloaded")
        await page.wait_for_timeout(500)

    await page.goto(f"{base}/remote?s={token}", wait_until="domcontentloaded")
    await page.wait_for_selector('[data-testid="remote-hero-card"]', timeout=20000)
    await page.wait_for_timeout(1000)

    garment_buttons = page.locator('[data-testid^="garment-card-"]')
    garment_count = await garment_buttons.count()
    for idx in range(garment_count):
        btn = garment_buttons.nth(idx)
        await btn.click(force=True)
        await page.wait_for_timeout(250)

    for fit_key in ["fitted", "regular", "relaxed"]:
        await page.click(f'[data-testid="fit-control-{fit_key}"]', force=True)
        await page.wait_for_timeout(250)

    for view_key in ["front", "back", "auto"]:
        await page.click(f'[data-testid="angle-view-{view_key}"]', force=True)
        await page.wait_for_timeout(250)

    await page.click('[data-testid="remove-garment-phone"]', force=True)
    await page.wait_for_timeout(600)
    await page.reload(wait_until="domcontentloaded")
    await page.wait_for_selector('[data-testid="remote-hero-card"]', timeout=20000)

    await page.goto(f"{base}/mirror?s={token}", wait_until="domcontentloaded")
    await page.wait_for_selector('[data-testid="cloth-toggle-mirror"]', timeout=30000)
    await page.wait_for_timeout(1200)
    await page.locator('[data-testid^="quick-garment-"]').first.click(force=True)
    await page.wait_for_timeout(1000)
    await page.click('[data-testid="take-snapshot-button"]', force=True)
    await page.wait_for_timeout(800)

    for width, height in [(320, 844), (768, 1024), (1024, 900), (1440, 900)]:
        await page.set_viewport_size({"width": width, "height": height})
        await page.goto(f"{base}/mirror?s={token}", wait_until="domcontentloaded")
        await page.wait_for_selector('[data-testid="model-status"]', timeout=30000)
        await page.wait_for_timeout(500)
        mirror_overflow = await page.evaluate("""() => ({
          doc_overflow: document.documentElement.scrollWidth > window.innerWidth,
          body_overflow: document.body.scrollWidth > window.innerWidth
        })""")
        print(f"Mirror viewport {width}: overflow={mirror_overflow}")

        await page.goto(f"{base}/remote?s={token}", wait_until="domcontentloaded")
        await page.wait_for_selector('[data-testid="remote-hero-card"]', timeout=30000)
        await page.wait_for_timeout(500)
        remote_overflow = await page.evaluate("""() => ({
          doc_overflow: document.documentElement.scrollWidth > window.innerWidth,
          body_overflow: document.body.scrollWidth > window.innerWidth
        })""")
        print(f"Remote viewport {width}: overflow={remote_overflow}")

    error_text = await page.evaluate("""() => {
    const errorElements = Array.from(document.querySelectorAll('.error, [class*="error"], [id*="error"]'));
    return errorElements.map(el => el.textContent).join(", ");
    }""")
    if error_text:
        print(f"Found error message: {error_text}")
    else:
        print("No error messages found on the page")

except Exception as e:
    print(f"❌ Test execution failed: {str(e)}")
'''

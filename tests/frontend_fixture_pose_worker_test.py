"""Playwright async script snippet for TEST-ONLY aura-pose worker fixture injection.

Usage: copy the `SCRIPT` string into mcp_browser_automation `script` field.
"""

SCRIPT = r'''
try:
    base = "https://ac6ee5bb-e30b-4d80-b303-523174d822e2.preview.emergentagent.com"
    await page.set_viewport_size({"width": 1920, "height": 1080})

    await page.add_init_script("""
      (() => {
        const fixtureLm = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, z: 0, visibility: 0.0 }));
        fixtureLm[11] = { x: 0.42, y: 0.34, z: 0, visibility: 0.99 };
        fixtureLm[12] = { x: 0.58, y: 0.34, z: 0, visibility: 0.99 };
        fixtureLm[23] = { x: 0.45, y: 0.58, z: 0, visibility: 0.99 };
        fixtureLm[24] = { x: 0.55, y: 0.58, z: 0, visibility: 0.99 };
        fixtureLm[13] = { x: 0.38, y: 0.47, z: 0, visibility: 0.99 };
        fixtureLm[14] = { x: 0.62, y: 0.47, z: 0, visibility: 0.99 };
        fixtureLm[15] = { x: 0.36, y: 0.60, z: 0, visibility: 0.99 };
        fixtureLm[16] = { x: 0.64, y: 0.60, z: 0, visibility: 0.99 };

        const fixtureWorld = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0 }));
        fixtureWorld[11] = { x: -0.2, y: -0.2, z: 0.0 };
        fixtureWorld[12] = { x: 0.2, y: -0.2, z: 0.0 };
        fixtureWorld[23] = { x: -0.16, y: 0.2, z: 0.0 };
        fixtureWorld[24] = { x: 0.16, y: 0.2, z: 0.0 };
        fixtureWorld[13] = { x: -0.28, y: -0.05, z: 0.02 };
        fixtureWorld[14] = { x: 0.28, y: -0.05, z: 0.02 };

        const RealWorker = window.Worker;
        window.Worker = function(url, options) {
          const workerName = options && options.name ? options.name : "";
          const urlStr = String(url || "");
          const shouldMock = workerName === "aura-pose" || urlStr.includes("poseWorker");
          if (!shouldMock) return new RealWorker(url, options);

          let onmessage = null;
          let onerror = null;
          return {
            postMessage(message) {
              try {
                if (message && message.type === "init") {
                  setTimeout(() => onmessage && onmessage({ data: { type: "ready" } }), 80);
                } else if (message && message.type === "frame") {
                  setTimeout(() => onmessage && onmessage({
                    data: { type: "result", landmarks: fixtureLm, worldLandmarks: fixtureWorld }
                  }), 16);
                }
              } catch (err) { if (onerror) onerror(err); }
            },
            terminate() {},
            addEventListener() {},
            removeEventListener() {},
            get onmessage() { return onmessage; },
            set onmessage(fn) { onmessage = fn; },
            get onerror() { return onerror; },
            set onerror(fn) { onerror = fn; }
          };
        };
      })();
    """)

    token = await page.evaluate("""async () => {
      const r = await fetch('/api/sessions', { method: 'POST' });
      const d = await r.json();
      const t = d.session.token;
      await fetch(`/api/sessions/${t}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ selected_garment: 'white_tee', cloth_enabled: true, source: 'phone' })
      });
      return t;
    }""")

    await page.goto(f"{base}/mirror?s={token}", wait_until="domcontentloaded")
    await page.wait_for_selector('[data-testid="model-status"]', timeout=30000)

    for _ in range(12):
      await page.wait_for_timeout(500)
      status = await page.locator('[data-testid="model-status"]').inner_text()
      pose_text = await page.locator('[data-testid="pose-counter"]').inner_text()
      cloth_text = await page.locator('[data-testid="cloth-status"]').inner_text()
      print(f"status={status} pose={pose_text} cloth={cloth_text}")
      if "READY" in status and "33/33" in pose_text and "DRAPE LIVE" in cloth_text:
        break

    await page.click('[data-testid="remove-garment-button"]', force=True)
    await page.wait_for_timeout(800)
    print(await page.locator('[data-testid="current-garment-name"]').inner_text())

    await page.locator('[data-testid^="quick-garment-"]').first.click(force=True)
    await page.wait_for_timeout(1000)
    await page.click('[data-testid="take-snapshot-button"]', force=True)
    await page.wait_for_timeout(800)

    error_text = await page.evaluate("""() => {
    const errorElements = Array.from(document.querySelectorAll('.error, [class*="error"], [id*="error"]'));
    return errorElements.map(el => el.textContent).join(", ");
    }""")
    if error_text:
        print(f"Found error message: {error_text}")
    else:
        print("No error messages found on the page")

except Exception as e:
    print(f"❌ Fixture test failed: {str(e)}")
'''

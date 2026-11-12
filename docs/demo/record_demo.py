"""Record a guided demo of the HeartSafe dashboard: one video (webm -> mp4) and a screenshot per step.

    pip install playwright imageio-ffmpeg && python -m playwright install chromium-headless-shell
    python record_demo.py http://localhost:8503 <out dir>

Drives a headless Chromium through Playwright. Run it against the demo copy (demo_app.py), because it
resolves a case, logs a night and a breathing session. A caption bar explains each step in the video;
it is hidden for the screenshots. Step 22 waits on the breathing timer, so the run takes about 3 minutes."""
import re
import subprocess
import sys
import time
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

URL, OUT = sys.argv[1], Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
VID = OUT / "_video"
W, H = 1440, 900
failed = []

CAPTION_JS = """(t) => {
  let el = document.getElementById('hs-demo-caption');
  if (!el) {
    el = document.createElement('div'); el.id = 'hs-demo-caption';
    el.style.cssText = 'position:fixed;left:50%;bottom:22px;transform:translateX(-50%);z-index:999999;' +
      'background:rgba(11,11,11,0.86);color:#fff;font:600 19px/1.35 "Segoe UI",system-ui,sans-serif;' +
      'padding:12px 22px;border-radius:12px;max-width:1100px;text-align:center;box-shadow:0 4px 18px rgba(0,0,0,.25)';
    document.body.appendChild(el);
  }
  el.style.display = t ? 'block' : 'none'; el.textContent = t || '';
}"""


def settle(page, extra=1.2):
    """Wait until Streamlit has finished running the script (status widget and cache spinners gone)."""
    time.sleep(0.4)
    for _ in range(480):
        if not (page.locator('[data-testid="stStatusWidget"]').count()
                or page.locator('[data-testid="stSpinner"]').count()):
            break
        time.sleep(0.25)
    time.sleep(extra)
    page.mouse.move(130, H - 40)  # park the pointer on the sidebar, so no chart tooltip shows


def caption(page, t):
    page.evaluate(CAPTION_JS, t)


def shot(page, name):
    caption(page, None)
    page.mouse.move(130, H - 40)
    time.sleep(0.15)
    page.screenshot(path=str(OUT / f"{name}.png"))


def scene(page, name, text, hold=3.0):
    """Show the caption, hold it for the video, then take the screenshot."""
    caption(page, text)
    time.sleep(hold)
    shot(page, name)
    caption(page, text)


def scroll(page, dy, steps=6):
    page.mouse.move(W * 0.6, H * 0.55)
    for _ in range(steps):
        page.mouse.wheel(0, dy / steps)
        time.sleep(0.08)
    time.sleep(0.6)


def top(page):
    scroll(page, -6000, 4)


def nav(page, title):
    page.locator('[data-testid="stSidebarNavLink"]').filter(has_text=title).first.click()
    settle(page)
    top(page)


def tab(page, name):
    page.get_by_role("tab", name=name).first.click()
    settle(page, 1.0)


def pick(page, label, option):
    page.locator('[data-testid="stSelectbox"]').filter(has_text=label).first.click()
    time.sleep(0.4)
    page.get_by_role("option", name=option).first.click()
    settle(page)


def step(fn):
    try:
        fn()
    except Exception as e:  # keep recording the other steps
        failed.append(f"{fn.__name__}: {str(e).splitlines()[0][:200]}")
        print("FAILED", failed[-1], flush=True)


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(viewport={"width": W, "height": H}, record_video_dir=str(VID),
                              record_video_size={"width": W, "height": H}, color_scheme="light")
    page = ctx.new_page()
    t_start = time.time()
    page.goto(URL, wait_until="networkidle")
    page.get_by_text("Why it was flagged").first.wait_for(timeout=240000)  # first load runs the follow-up watch
    settle(page, 2)
    skip = max(0.0, time.time() - t_start - 1.0)  # cut the loading spinner from the start of the video

    def s01_queue():
        scene(page, "01-clinician-queue", "1 · Clinician queue: every open case, most urgent first, with the "
              "follow-up watch that re-checks deadlines every 2 hours", 4.5)

    def s02_case():
        page.get_by_text("Why it was flagged").first.scroll_into_view_if_needed()
        scroll(page, -120, 2)
        scene(page, "02-case-review", "2 · An explainable review: why it was flagged, symptoms, ECG and MRI reports "
              "in plain words, data-quality notes and a suggested next step", 5)

    def s03_followup():
        tab(page, "Follow-up")
        scene(page, "03-case-followup", "3 · Follow-up: deadlines, reminders, and the patient loop after the "
              "clinician decides", 3.5)
        tab(page, "Audit log")
        scene(page, "04-case-audit", "4 · Audit log: every action, who did it and when (append-only)", 3)
        tab(page, "Review")

    def s04_ack():
        btn = page.get_by_role("button", name=re.compile(r"^(Acknowledge|Review|Resolve) case"))
        btn.first.scroll_into_view_if_needed()
        caption(page, "5 · Only a clinician moves a case: acknowledge → review → resolve (with a note)")
        time.sleep(2)
        note = page.get_by_label("Resolution note (required)")
        if note.count():
            note.first.click()
            note.first.type("Called the patient; plan per the heart-failure protocol.", delay=25)
            time.sleep(0.8)
        btn.first.click()
        settle(page, 1.5)
        shot(page, "05-case-acknowledged")

    def s05_journey():
        nav(page, "Patient journey")
        scene(page, "06-patient-journey", "6 · Patient journey (demo patient, 68, remote community): daily "
              "readings with every day HeartSafe acted", 4.5)
        page.get_by_text("Alerts and HeartSafe actions").first.scroll_into_view_if_needed()
        scroll(page, -80, 2)
        scene(page, "07-journey-actions", "7 · What HeartSafe did with each alert: a new scale verified by a nurse, "
              "unusable ECGs re-recorded, a sleep check-in, a tracked clinical case", 5)
        page.get_by_text(re.compile(r"^Cases \(\d+\)")).first.scroll_into_view_if_needed()
        scene(page, "08-journey-cases", "8 · Every case for this patient, with its review and follow-up", 3)

    def s06_equity():
        nav(page, "Equity audit")
        scene(page, "09-equity-gaps", "9 · Equity audit: is any group (remoteness, sex, age, device) served worse? "
              "Bootstrap 95% confidence intervals", 4.5)
        tab(page, "Safeguards")
        scene(page, "10-equity-safeguards", "10 · Gaps become safeguard proposals with their workload cost; only an "
              "administrator can approve them", 4)

    def s07_results():
        nav(page, "Results")
        scene(page, "11-results", "11 · HeartSafe vs the platform alone on the same simulated patients: faster "
              "acknowledgement, faulty-reading alerts stopped", 4.5)
        tab(page, "Patient loop")
        scene(page, "12-results-patient-loop", "12 · Patient loop: instructions completed, helpers contacted when "
              "the patient does not answer", 3.5)
        tab(page, "Heart rhythm")
        scene(page, "13-results-rhythm", "13 · Heart rhythm: new atrial fibrillation reported, known AF only "
              "logged, heart rate checked against the patient's own limits", 5)
        tab(page, "Sleep and breathing")
        scene(page, "14-results-sleep", "14 · Sleep and breathing: routine check-in calls, never a clinical alert "
              "on their own", 4)

    def s08_models():
        nav(page, "Model reliability")
        scene(page, "15-models-ecg", "15 · Model reliability: ECG classifier on held-out data and on a new device; "
              "injected faults caught", 4)
        page.get_by_text("Naming the rhythm").first.scroll_into_view_if_needed()
        scroll(page, -60, 2)
        scene(page, "16-models-rhythm", "16 · The rhythm model (normal / AF / other) and heart rate from the "
              "R-peaks", 4)
        top(page)
        tab(page, "Cardiac MRI · M&Ms")
        scene(page, "17-models-mri", "17 · Cardiac MRI segmentation, including scanners never seen in training", 4)

    def s09_how():
        nav(page, "How it works")
        scene(page, "18-how-it-works", "18 · How HeartSafe works, in seven steps", 4)

    def s10_patient_sleep():
        pick(page, "Signed in as", "Patient")
        top(page)
        scene(page, "19-patient-sleep", "19 · The patient's own page: log last night's sleep, compared with "
              "their own usual night", 4)
        for label_, value in (("Hours slept", "4"), ("Times you woke up", "3")):
            box = page.locator('[data-testid="stNumberInput"]').filter(has_text=label_).locator("input").first
            box.fill(value)
            box.press("Tab")
            time.sleep(0.5)
        page.get_by_text("I woke up short of breath").first.click()
        caption(page, "20 · She slept 4 hours, woke 3 times and woke up short of breath")
        time.sleep(2.5)
        page.get_by_role("button", name="Save last night's sleep").click()
        settle(page, 1.5)
        top(page)
        scene(page, "20-patient-sleep-saved", "20 · Breathlessness at night is a symptom the platform alerts on: "
              "the care team is told and a nurse is asked to call", 5)

    def s11_breathing():
        tab(page, "Breathing exercises")
        scene(page, "21-breathing", "21 · Breathing exercises from the care team's plan, with a paced timer", 3)
        fr = page.frame_locator("iframe").first
        fr.get_by_role("button", name="Start", exact=True).click()
        caption(page, "22 · The circle grows on the breath in and shrinks on the breath out")
        time.sleep(11)
        shot(page, "22-breathing-timer")
        caption(page, "22 · The circle grows on the breath in and shrinks on the breath out")
        time.sleep(1)
        page.get_by_role("button", name="I finished this session").click()
        settle(page, 1.5)
        top(page)
        scene(page, "23-breathing-logged", "23 · Each finished session is counted against the plan "
              "(3 a day by default)", 3.5)
        tab(page, "My entries")
        scene(page, "24-my-entries", "24 · Everything the patient logged; only the patient side can add entries", 3)

    def s12_back_to_clinician():
        pick(page, "Signed in as", "Clinician")
        nav(page, "Patient journey")
        page.get_by_text("Sleep and breathing (logged by the patient)").first.scroll_into_view_if_needed()
        scroll(page, -80, 2)
        scene(page, "25-clinician-sees-sleep", "25 · The care team sees the patient's sleep and breathing next to "
              "the other readings and in every case review", 5)
        caption(page, "HeartSafe checks, explains and follows up. The clinical team makes every decision.")
        time.sleep(4)

    for fn in (s01_queue, s02_case, s03_followup, s04_ack, s05_journey, s06_equity, s07_results, s08_models,
               s09_how, s10_patient_sleep, s11_breathing, s12_back_to_clinician):
        step(fn)
        print("done", fn.__name__, flush=True)
    webm = page.video.path()
    ctx.close()
    browser.close()

mp4 = OUT / "heartsafe-demo.mp4"
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{skip:.1f}", "-i", str(webm),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-crf", "26", "-preset", "slow", "-movflags", "+faststart", str(mp4)], check=True)
Path(webm).unlink()
VID.rmdir()
print("video:", mp4, f"{mp4.stat().st_size / 1e6:.1f} MB")
print("FAILED STEPS:", failed if failed else "none")

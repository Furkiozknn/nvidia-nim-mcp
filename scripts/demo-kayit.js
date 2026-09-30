/*
 * demo-kayit.js - records the terminal pages written by demo-uret.py.
 *
 *   node scripts/demo-kayit.js <docs/demo>
 *
 * Needs playwright (npm i -g playwright; npx playwright install chromium) and
 * ffmpeg on PATH. Output: demo.mp4 + demo.gif (1280x720 landscape, README) and
 * demo-dikey.mp4 (1080x1920, no audio, no captions: raw material for a short video).
 */
const fs = require("fs");
const path = require("path");
const cp = require("child_process");

let pw;
try { pw = require("playwright"); }
catch (e) {
  const g = cp.spawnSync("npm", ["root", "-g"], { encoding: "utf8", shell: true }).stdout.trim();
  pw = require(path.join(g, "playwright"));
}

const OUT = path.resolve(process.argv[2] || path.join(__dirname, "..", "docs", "demo"));

function ff(args) {
  const r = cp.spawnSync("ffmpeg", ["-y", "-loglevel", "error", ...args], { encoding: "utf8" });
  if (r.status) throw new Error(r.stderr);
}

async function kaydet(browser, sayfa, sorgu, w, h) {
  const ctx = await browser.newContext({ viewport: { width: w, height: h }, recordVideo: { dir: OUT, size: { width: w, height: h } } });
  const page = await ctx.newPage();
  await page.goto("file:///" + path.join(OUT, sayfa).replace(/\\/g, "/") + sorgu);
  await page.evaluate(() => document.fonts.ready);
  const bitis = await page.evaluate(() => window.__bitis);
  await page.waitForTimeout(bitis + 600);
  const video = page.video();
  await ctx.close();
  return video.path();
}

(async () => {
  const browser = await pw.chromium.launch();

  const yatay = await kaydet(browser, "demo.html", "", 1280, 720);
  ff(["-i", yatay, "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path.join(OUT, "demo.mp4")]);
  ff(["-i", yatay, "-vf", "fps=6,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=24:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle", path.join(OUT, "demo.gif")]);
  fs.unlinkSync(yatay);

  const dikey = await kaydet(browser, "demo.html", "?dikey", 1080, 1920);
  ff(["-i", dikey, "-an", "-vf", "fps=30", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", path.join(OUT, "demo-dikey.mp4")]);
  fs.unlinkSync(dikey);

  await browser.close();
  for (const f of ["demo.mp4", "demo.gif", "demo-dikey.mp4"]) {
    console.log(f, (fs.statSync(path.join(OUT, f)).size / 1024).toFixed(0) + " KB");
  }
})();

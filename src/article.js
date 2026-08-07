import { marked } from "marked";
import articleMarkdown from "../ARTICLE.md?raw";
import "./article.css";

marked.use({ gfm: true, breaks: false });

const article = document.querySelector("#article");
const withoutDuplicateTitle = articleMarkdown.replace(/^# .+\n+/, "");
article.innerHTML = marked.parse(withoutDuplicateTitle);

for (const code of article.querySelectorAll("pre code.language-mermaid")) {
  const diagram = document.createElement("div");
  diagram.className = "architecture-diagram";
  diagram.setAttribute("role", "img");
  diagram.setAttribute("aria-label", "Three-channel temporal input passes through MobileNetV4 and a feature pyramid to produce a hotspot heatmap and subpixel offsets.");
  diagram.innerHTML = `
    <div class="architecture-stage input-stage"><span>Input</span><strong>3 × 640 × 640</strong><small>grayscale + 2 motion channels</small></div>
    <span class="architecture-arrow">→</span>
    <div class="architecture-stage"><span>Backbone</span><strong>MobileNetV4</strong><small>Conv-Small</small></div>
    <span class="architecture-arrow">→</span>
    <div class="architecture-stage"><span>Fusion</span><strong>Feature pyramid</strong><small>strides 4 · 8 · 16</small></div>
    <span class="architecture-arrow">→</span>
    <div class="architecture-outputs">
      <div class="architecture-stage output-stage"><span>Heatmap</span><strong>1 × 160 × 160</strong></div>
      <div class="architecture-stage output-stage"><span>Offsets</span><strong>2 × 160 × 160</strong></div>
    </div>`;
  code.parentElement.replaceWith(diagram);
}

import { marked } from "marked";
import mermaid from "mermaid";
import articleMarkdown from "../ARTICLE.md?raw";
import "./article.css";

marked.use({ gfm: true, breaks: false });

const article = document.querySelector("#article");
const withoutDuplicateTitle = articleMarkdown.replace(/^# .+\n+/, "");
article.innerHTML = marked.parse(withoutDuplicateTitle);

for (const code of article.querySelectorAll("pre code.language-mermaid")) {
  const diagram = document.createElement("div");
  diagram.className = "mermaid";
  diagram.textContent = code.textContent;
  code.parentElement.replaceWith(diagram);
}

mermaid.initialize({
  startOnLoad: false,
  theme: "neutral",
  securityLevel: "strict",
  fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif"
});
await mermaid.run({ querySelector: ".mermaid" });

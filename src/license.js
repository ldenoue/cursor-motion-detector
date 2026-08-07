import { marked } from "marked";
import licenseMarkdown from "../LICENSE.md?raw";
import "./article.css";

marked.use({ gfm: true, breaks: false });

const license = document.querySelector("#license");
const withoutDuplicateTitle = licenseMarkdown.replace(/^# .+\n+/, "");
license.innerHTML = marked.parse(withoutDuplicateTitle);

// Markdown to HTML, for the agent's replies in the chat (js/chat.js). markdown-it does the work
// (vendored as is); this file only sets it up:
// - raw HTML is off: markup a model writes shows as text
// - markdown-it's own link check drops javascript:, vbscript: and the like
// - tables and strikethrough are built in; a single newline is a line break, as the model wrote it
// - images are off: a reply never makes the app load something from the internet
// - links to other sites open in a new window (pywebview: the system browser); app paths stay
import "../vendor/markdown-it-14.1.0.min.js";

const md = globalThis.markdownit({ html: false, linkify: true, breaks: true }).disable("image");

const outside = /^[a-z][a-z\d+.-]*:/i; // has a scheme: https:, mailto:…

md.renderer.rules.link_open = (tokens, i, options, env, self) => {
  if (outside.test(tokens[i].attrGet("href") ?? "")) {
    tokens[i].attrSet("target", "_blank");
    tokens[i].attrSet("rel", "noopener");
  }
  return self.renderToken(tokens, i, options);
};

export function render(markdown) {
  return md.render(markdown);
}

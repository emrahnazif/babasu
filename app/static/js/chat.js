// Agent chat in the sidebar. Not connected to an agent yet:
// sending shows your message and a placeholder reply.
const form = document.querySelector(".chat__form");

if (form) {
  const messages = document.querySelector(".chat__messages");
  const empty = document.querySelector(".chat__empty");
  const input = form.elements.message;

  // Your own message always scrolls into view; a reply follows only if you are
  // already at the bottom, so it doesn't pull you away from what you are reading.
  function addMessage(role, text) {
    const atBottom = messages.scrollHeight - messages.scrollTop - messages.clientHeight < 8;
    const message = document.createElement("div");
    message.className = `chat__message chat__message--${role}`;
    message.textContent = text;
    messages.append(message);
    empty.hidden = true;
    if (role === "user" || atBottom) messages.scrollTop = messages.scrollHeight;
  }

  // Enter sends, Shift+Enter adds a new line (isComposing: don't send mid-IME input).
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    addMessage("user", text);
    addMessage("assistant", "Not connected yet.");
  });
}

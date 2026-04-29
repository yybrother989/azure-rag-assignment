// header_files_button.js
//
// Bridges the header link declared in .chainlit/config.toml's [[UI.header_links]]
// (display_name="📁 Files", url="#open-files") to the Chainlit chat input,
// so a single click on that header button effectively types `files` into the
// composer and submits — which the Python @cl.on_message handler intercepts
// to refresh the right-side file-tree sidebar.
//
// Why this hack: Chainlit's [[UI.header_links]] only supports static URLs;
// it doesn't expose a Python action callback. To trigger a backend action
// from the header we have to round-trip through the actual chat input.
//
// Side effect: a one-line "files" user message appears in chat history per
// click. Acceptable trade-off vs writing a full custom React frontend.

(function () {
  "use strict";

  const TRIGGER_HREF = "#open-files";
  const COMMAND_TEXT = "files";

  function getInputTextarea() {
    return (
      document.getElementById("chat-input") ||
      document.querySelector('textarea[placeholder*="message" i]') ||
      document.querySelector('textarea[name="message"]') ||
      document.querySelector("form textarea")
    );
  }

  function getSendButton() {
    // Chainlit's submit button is typically the last button inside the form
    // around the textarea, with type="submit" or an aria-label like "Send".
    return (
      document.querySelector('button[id="chat-submit"]') ||
      document.querySelector('button[aria-label="Send"i]') ||
      document.querySelector('button[type="submit"]') ||
      // Fallback: the icon button right after the textarea
      (() => {
        const ta = getInputTextarea();
        if (!ta) return null;
        const form = ta.closest("form");
        if (!form) return null;
        const buttons = form.querySelectorAll("button");
        return buttons[buttons.length - 1] || null;
      })()
    );
  }

  // React-controlled inputs ignore plain `el.value = ...`. Use the prototype
  // setter so React's onChange registers the new value.
  function setReactValue(el, value) {
    const proto = Object.getPrototypeOf(el);
    const valueSetter = Object.getOwnPropertyDescriptor(proto, "value")?.set;
    if (valueSetter) {
      valueSetter.call(el, value);
    } else {
      el.value = value;
    }
    el.dispatchEvent(new Event("input", { bubbles: true }));
  }

  async function triggerFilesCommand() {
    const ta = getInputTextarea();
    if (!ta) {
      console.warn("[Files button] could not find chat input textarea");
      return;
    }
    setReactValue(ta, COMMAND_TEXT);
    // Wait one frame so React processes the state change before we click submit
    await new Promise((r) => requestAnimationFrame(r));
    const btn = getSendButton();
    if (!btn) {
      console.warn("[Files button] could not find send button — pressing Enter");
      ta.focus();
      ta.dispatchEvent(
        new KeyboardEvent("keydown", { key: "Enter", code: "Enter", bubbles: true })
      );
      return;
    }
    btn.click();
  }

  function isOurHeaderLink(target) {
    if (!target) return false;
    // Walk up to the anchor since the click might land on inner span/svg
    const anchor = target.closest && target.closest("a[href]");
    if (!anchor) return false;
    return anchor.getAttribute("href") === TRIGGER_HREF;
  }

  // Intercept clicks on the header link before the browser navigates to the
  // hash. capture=true so we run before React's own handlers.
  document.addEventListener(
    "click",
    function (event) {
      if (!isOurHeaderLink(event.target)) return;
      event.preventDefault();
      event.stopPropagation();
      triggerFilesCommand().catch((e) => console.error("[Files button]", e));
    },
    true
  );

  // Also handle hashchange in case the click slips through (e.g. middle-click).
  window.addEventListener("hashchange", function () {
    if (window.location.hash === TRIGGER_HREF) {
      // Reset the hash so a second click of the same link still fires hashchange.
      history.replaceState(null, "", window.location.pathname + window.location.search);
      triggerFilesCommand().catch((e) => console.error("[Files button]", e));
    }
  });
})();

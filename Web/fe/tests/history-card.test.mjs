import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("history renderer forwards persisted assistant cards", async () => {
  const html = await readFile(new URL("../index.html", import.meta.url), "utf8");

  assert.match(
    html,
    /messages\.forEach\(\(message\)\s*=>\s*appendMessage\(message,\s*\{\s*assistantCard:\s*message\.assistant_card\s*\|\|\s*null\s*\}\)\)/,
  );
});

test("selection reply uses a separate composer strip and request field", async () => {
  const html = await readFile(new URL("../index.html", import.meta.url), "utf8");
  const preloadStart = html.indexOf("function preloadSelectionPrompt()");
  const preloadEnd = html.indexOf("function messageShell", preloadStart);
  const preloadFunction = html.slice(preloadStart, preloadEnd);

  assert.match(html, /id="reply-context"/);
  assert.match(html, /id="reply-context-quote"/);
  assert.match(html, /id="reply-context-close"/);
  assert.doesNotMatch(preloadFunction, /textarea\.value\s*=/);
  assert.match(preloadFunction, /setPendingReplyContext/);
  assert.match(html, /maxReplyQuoteLength\s*=\s*4000/);
  assert.match(html, /reply_context:\s*replyContext/);
  assert.match(html, /message\.reply_context/);
});

test("assistant selection and active timeline use readable theme colors", async () => {
  const html = await readFile(new URL("../index.html", import.meta.url), "utf8");

  assert.match(html, /\.assistant-selectable::selection/);
  assert.match(html, /data-timeline-text/);
  assert.match(html, /classList\.toggle\("text-white",\s*isActive\)/);
});

test("assistant media from a general response is rendered and reloadable", async () => {
  const html = await readFile(new URL("../index.html", import.meta.url), "utf8");

  assert.match(html, /message\.media/);
  assert.match(html, /loading="lazy"/);
  assert.match(html, /assistant-media/);
});

test("assistant responses render markdown emphasis, lists, and safe links", async () => {
  const html = await readFile(new URL("../index.html", import.meta.url), "utf8");

  assert.match(html, /function renderAssistantText\(text\)/);
  assert.match(html, /<strong>\$1<\/strong>/);
  assert.match(html, /renderAssistantText\(message\.content\)/);
  assert.match(html, /renderAssistantText\(card\.summary\)/);
  assert.match(html, /Nhận định tổng hợp/);
  assert.match(html, /card\.interpretation/);
});

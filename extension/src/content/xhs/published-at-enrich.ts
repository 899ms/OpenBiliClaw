/**
 * Fill exact Xiaohongshu publish times from a hydrated note page.
 *
 * Search / collect cards expose no time field, so when the backend flags a
 * non-"all" date preference the task executor loads up to five note URLs in
 * hidden same-origin iframes and reads
 * `__INITIAL_STATE__.note.noteDetailMap[noteId].note.time` (epoch ms). The
 * iframe is removed as soon as the value is read (or on timeout); failures are
 * best-effort and never fail the task.
 */

import {
  extractNoteIdFromUrl,
  extractPublishedAtFromState,
} from "../../shared/xhs-published-at.ts";

const POLL_INTERVAL_MS = 300;
const MAX_CONCURRENCY = 3;

export type PublishedAtLoader = (
  url: string,
  noteId: string,
  timeoutMs: number,
) => Promise<number | undefined>;

export interface EnrichPublishedAtOptions {
  maxNotes?: number;
  timeoutMs?: number;
  concurrency?: number;
  document?: Document;
  loader?: PublishedAtLoader;
}

export function isXhsNoteUrl(rawUrl: string): boolean {
  if (!rawUrl) return false;
  try {
    const parsed = new URL(rawUrl, "https://www.xiaohongshu.com");
    if (!parsed.hostname.endsWith("xiaohongshu.com")) return false;
    return Boolean(extractNoteIdFromUrl(parsed.href));
  } catch {
    return false;
  }
}

/** Load one note in a hidden iframe and read its exact epoch-ms publish time. */
export function loadPublishedAtFromIframe(
  url: string,
  noteId: string,
  timeoutMs: number,
  doc: Document = document,
): Promise<number | undefined> {
  return new Promise((resolve) => {
    const frame = doc.createElement("iframe");
    frame.setAttribute("aria-hidden", "true");
    frame.style.cssText = "width:1px;height:1px;opacity:0;position:fixed;left:-9999px;top:-9999px";

    let settled = false;
    const finish = (value?: number): void => {
      if (settled) return;
      settled = true;
      clearInterval(pollTimer);
      clearTimeout(timeoutTimer);
      try {
        frame.remove();
      } catch {
        // removal is best effort
      }
      resolve(value);
    };

    const timeoutTimer = setTimeout(() => finish(undefined), timeoutMs);
    const pollTimer = setInterval(() => {
      try {
        const frameWindow = frame.contentWindow as
          | (Window & { __INITIAL_STATE__?: unknown })
          | null;
        if (!frameWindow) return;
        const publishedAt = extractPublishedAtFromState(frameWindow.__INITIAL_STATE__, noteId);
        if (publishedAt !== undefined) finish(publishedAt);
      } catch {
        // Cross-origin or mid-navigation; keep polling until the timeout.
      }
    }, POLL_INTERVAL_MS);

    try {
      frame.src = url;
      (doc.body ?? doc.documentElement).appendChild(frame);
    } catch {
      finish(undefined);
    }
  });
}

/** Enrich notes concurrently under a hard note-count ceiling. */
export async function enrichNotesWithPublishedAt<
  T extends { url?: string; published_at?: string | number },
>(notes: T[], options: EnrichPublishedAtOptions = {}): Promise<number> {
  const maxNotes = Math.max(0, options.maxNotes ?? 5);
  const timeoutMs = Math.max(1_000, options.timeoutMs ?? 6_000);
  const concurrency = Math.max(1, Math.min(MAX_CONCURRENCY, options.concurrency ?? 2));
  const loader =
    options.loader ??
    ((url: string, noteId: string, timeout: number): Promise<number | undefined> =>
      loadPublishedAtFromIframe(url, noteId, timeout, options.document));

  const targets = notes
    .filter((note) => !note.published_at && typeof note.url === "string" && isXhsNoteUrl(note.url))
    .slice(0, maxNotes);
  if (targets.length === 0) return 0;

  let cursor = 0;
  let enriched = 0;
  const worker = async (): Promise<void> => {
    while (cursor < targets.length) {
      const note = targets[cursor];
      cursor += 1;
      const noteId = extractNoteIdFromUrl(note.url ?? "");
      if (!noteId) continue;
      const publishedAt = await loader(note.url as string, noteId, timeoutMs).catch(
        () => undefined,
      );
      if (publishedAt === undefined) continue;
      note.published_at = publishedAt;
      enriched += 1;
    }
  };
  await Promise.all(
    Array.from({ length: Math.min(concurrency, targets.length) }, () => worker()),
  );
  return enriched;
}

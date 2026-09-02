/**
 * Clean up the LLM's markdown before rendering.
 *
 * `react-markdown` deliberately does not render raw HTML, so any tag the
 * model emits would otherwise show up literally as `<br>` in the answer.
 * The generation prompt now forbids HTML, but models drift, and a visible
 * `<br>` is a bad enough result to be worth handling here too.
 */

const BR = /<br\s*\/?>/gi;
const HTML_TAG = /<\/?(?:b|strong|i|em|u|span|div|p)\s*\/?>/gi;

export function normalizeMarkdown(input: string): string {
  return input
    .split("\n")
    .map((line) => {
      const isTableRow = line.trimStart().startsWith("|");

      // A GFM table cell cannot contain a newline — turning <br> into one
      // inside a row would break the table apart. Use a separator there
      // and a real line break everywhere else.
      let cleaned = isTableRow ? line.replace(BR, " · ") : line.replace(BR, "\n");
      cleaned = cleaned.replace(HTML_TAG, "");

      // The model likes "• " for bullets, which markdown doesn't know
      // about — it renders as a literal dot with no list formatting.
      if (!isTableRow) {
        cleaned = cleaned.replace(/^(\s*)•\s+/, "$1- ");
      }
      return cleaned;
    })
    .join("\n");
}
